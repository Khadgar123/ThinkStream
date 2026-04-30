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
    """HF Trainer subclass with audit logging + per-class eval metrics.

    Vanilla CE on assistant tokens (DeepEyesV2 / Qwen-VL official convention).
    No per-class loss reweighting, no class-balanced sampler — uniform on
    the trajectory's natural sample distribution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit_step = self._init_audit_writers()
        self._reset_eval_accumulator()
        self._reset_train_metrics()

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

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Vanilla CE on assistant tokens (DeepEyesV2 / Qwen-VL official
        # convention). No per-class re-weighting; data is balanced enough
        # that uniform weight + standard sampler is sufficient.
        sample_weights = inputs.pop("sample_weights", None)  # ignored (always 1.0)
        token_loss_weight = inputs.pop("token_loss_weight", None)  # ignored (uniform)
        sample_meta = inputs.pop("sample_meta", None)
        eval_meta = inputs.pop("eval_meta", None)
        # Keep input_ids handy for argmax-vs-gold accumulation during eval
        eval_input_ids = inputs["input_ids"] if not self.model.training else None

        outputs = model(**inputs)
        loss = outputs.loss
        per_sample_loss_for_audit = None

        # ── Eval-time accuracy accumulation (teacher-forced argmax) ──
        # Done before audit because audit guard requires model.training=True
        # and this branch is the eval path.
        if not self.model.training and eval_meta is not None and eval_input_ids is not None:
            try:
                self._accumulate_eval_argmax(outputs.logits, eval_input_ids, eval_meta)
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).debug("eval argmax accum skipped: %s", e)

        # ── Train-time per-class metrics accumulation (flushed in log()) ──
        # Tracks per-action-type loss, sample weight, and teacher-forced
        # action argmax accuracy so wandb shows whether class balancing is
        # working AND whether the action keyword is being predicted
        # correctly per class (catches compress collapse early).
        if self.model.training and sample_meta:
            try:
                self._accumulate_train_metrics(
                    per_sample_loss=per_sample_loss_for_audit,
                    sample_weights=sample_weights,
                    sample_meta=sample_meta,
                    eval_meta=eval_meta,
                    logits=outputs.logits if eval_meta is not None else None,
                    input_ids=inputs.get("input_ids") if eval_meta is not None else None,
                )
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).debug("train metrics accum skipped: %s", e)

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
    # Eval-time argmax accuracy (teacher-forced).
    #   - v12_argmax_match / v12_argmax_total : holistic argmax over the
    #     full assistant span [ans_start, ans_end]. Per-class breakdown by
    #     sample_type. (v11's per-position metrics on <action>/<summary>/
    #     <query>/<response> are gone — v12 has no fixed structural spans.)
    # -----------------------------------------------------------------

    def _reset_eval_accumulator(self):
        self._eval_acc = {
            "v12_argmax_match":  defaultdict(int),
            "v12_argmax_total":  defaultdict(int),
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


                # v12.0: holistic teacher-forced argmax over the FULL
                # assistant span. Replaces v11's per-position structural
                # metrics (action_keyword / summary / query / response /
                # format_compliance) which all return 0 in v12 because
                # data_processor sets those positions to []. This single
                # signal answers: "given the gold prefix, how often does
                # the model's argmax match the gold next token across
                # every assistant token?". Free-gen gate is the primary
                # behavioural eval (scripts/eval/v12_freegen_gate.py).
                ans_start = meta.get("ans_start")
                ans_end = meta.get("ans_end")
                if ans_start is not None and ans_end is not None:
                    s = max(1, int(ans_start))   # logits[p-1] predicts pos p
                    e = min(L, int(ans_end) + 1)
                    matched = 0
                    total = 0
                    for p in range(s, e):
                        gold = int(input_ids[b, p].item())
                        if preds[b, p - 1].item() == gold:
                            matched += 1
                        total += 1
                    if total > 0:
                        self._eval_acc["v12_argmax_total"][stype] += total
                        self._eval_acc["v12_argmax_total"]["_all"] += total
                        self._eval_acc["v12_argmax_match"][stype] += matched
                        self._eval_acc["v12_argmax_match"]["_all"] += matched

                    # v12.1 BEHAVIORAL METRICS — decode argmax tokens →
                    # parse v12 protocol → emit kind/format counters per
                    # sample_type. Lets us see at SFT time:
                    #   "did silent samples emit empty <answer>?"
                    #   "did compress samples emit a compress tool_call?"
                    #   "did the model recall when expected to recall?"
                    # answers free-gen gate's questions WITHOUT vLLM
                    # rollout (only single forward pass already done).
                    self._accumulate_v12_behavioral(
                        preds, input_ids, b, s, e, stype,
                    )

    def _accumulate_v12_behavioral(
        self, preds, input_ids, b: int, s: int, e: int, stype: str,
    ) -> None:
        """v12.1 per-sample behavioral counters from teacher-forced argmax."""
        # Lazy-init on first call so __init__ doesn't change.
        if "v12_kind_match" not in self._eval_acc:
            for k in (
                "v12_kind_match", "v12_kind_total",
                "v12_format_valid", "v12_format_total",
                "v12_observed_recall", "v12_observed_compress",
                "v12_observed_answer",  "v12_observed_unknown",
                "v12_silent_empty_match", "v12_answer_nonempty",
            ):
                self._eval_acc[k] = defaultdict(int)

        try:
            tokenizer = (
                getattr(self, "processing_class", None)
                or getattr(self, "tokenizer", None)
            )
            if tokenizer is None or not hasattr(tokenizer, "decode"):
                return
            argmax_ids = preds[b, s - 1: e - 1].tolist()
            decoded = tokenizer.decode(argmax_ids, skip_special_tokens=False)
        except Exception:
            return

        from thinkstream.data.agent_protocol import parse_agent_output_v12
        parsed = parse_agent_output_v12(decoded)
        observed_kind = parsed.get("kind", "unknown")
        format_valid = parsed.get("format_error") is None

        self._eval_acc["v12_format_total"][stype] += 1
        self._eval_acc["v12_format_total"]["_all"] += 1
        if format_valid:
            self._eval_acc["v12_format_valid"][stype] += 1
            self._eval_acc["v12_format_valid"]["_all"] += 1

        if stype == "silent":
            expected_kind = "answer_empty"
        elif stype in ("response", "recall_response"):
            expected_kind = "answer_nonempty"
        elif stype in ("recall_query", "recall"):
            expected_kind = "recall"
        elif stype in ("compress", "compress_inter"):
            expected_kind = "compress"
        else:
            expected_kind = "unknown"

        observed_bucket = {
            "answer": "v12_observed_answer",
            "recall": "v12_observed_recall",
            "compress": "v12_observed_compress",
            "unknown": "v12_observed_unknown",
        }.get(observed_kind, "v12_observed_unknown")
        self._eval_acc[observed_bucket][stype] += 1
        self._eval_acc[observed_bucket]["_all"] += 1

        self._eval_acc["v12_kind_total"][stype] += 1
        self._eval_acc["v12_kind_total"]["_all"] += 1
        kind_match = False
        if expected_kind == "answer_empty":
            kind_match = (
                observed_kind == "answer"
                and (parsed.get("answer_text") or "") == ""
            )
            if observed_kind == "answer" and (parsed.get("answer_text") or "") == "":
                self._eval_acc["v12_silent_empty_match"][stype] += 1
                self._eval_acc["v12_silent_empty_match"]["_all"] += 1
        elif expected_kind == "answer_nonempty":
            kind_match = (
                observed_kind == "answer"
                and bool(parsed.get("answer_text"))
            )
            if observed_kind == "answer" and parsed.get("answer_text"):
                self._eval_acc["v12_answer_nonempty"][stype] += 1
                self._eval_acc["v12_answer_nonempty"]["_all"] += 1
        elif expected_kind in ("recall", "compress"):
            kind_match = (observed_kind == expected_kind)
        if kind_match:
            self._eval_acc["v12_kind_match"][stype] += 1
            self._eval_acc["v12_kind_match"]["_all"] += 1

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
        # Pack 12 counters × len(all_keys) into one tensor for a single all_reduce
        n = len(all_keys)
        bands = [
            "v12_argmax_match", "v12_argmax_total",
            # v12.1 behavioral metrics
            "v12_kind_match", "v12_kind_total",
            "v12_format_valid", "v12_format_total",
            "v12_observed_recall", "v12_observed_compress",
            "v12_observed_answer", "v12_observed_unknown",
            "v12_silent_empty_match", "v12_answer_nonempty",
        ]
        buf = torch.zeros(len(bands) * n, dtype=torch.long, device=device)
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
        # v12.0: holistic assistant-span argmax accuracy.
        v12_tot = self._eval_acc["v12_argmax_total"].get("_all", 0)
        if v12_tot > 0:
            out["eval/v12_assistant_argmax_acc"] = (
                self._eval_acc["v12_argmax_match"].get("_all", 0) / v12_tot
            )
        for stype, tot in self._eval_acc["v12_argmax_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/v12_assistant_argmax_acc_{stype}"] = (
                self._eval_acc["v12_argmax_match"].get(stype, 0) / tot
            )

        # v12.1 BEHAVIORAL METRICS (parsed from teacher-forced argmax)
        # — answers user-actionable questions about model behavior:
        #   "Does silent emit empty <answer>?"  → v12_silent_empty_rate_silent
        #   "Does compress emit compress tool_call?" → v12_compress_emit_rate_compress
        #   "Does recall emit recall tool_call?"   → v12_recall_emit_rate_recall_query
        #   "Does response emit non-empty <answer>?" → v12_answer_emit_rate_response
        #   "Is the parsed format valid?"           → v12_format_valid_<stype>
        #   "Did the model pick the right kind?"    → v12_kind_match_<stype>
        kind_tot = self._eval_acc.get("v12_kind_total", {}).get("_all", 0)
        if kind_tot > 0:
            out["eval/v12_kind_match"] = (
                self._eval_acc["v12_kind_match"].get("_all", 0) / kind_tot
            )
            out["eval/v12_format_valid"] = (
                self._eval_acc["v12_format_valid"].get("_all", 0) / kind_tot
            )
            # Confusion: how often does the model emit each kind regardless
            # of expectation. Helps catch silent collapse / over-recall etc.
            for obs_band, name in [
                ("v12_observed_answer", "answer_emit_rate"),
                ("v12_observed_recall", "recall_emit_rate"),
                ("v12_observed_compress", "compress_emit_rate"),
            ]:
                out[f"eval/v12_{name}"] = (
                    self._eval_acc.get(obs_band, {}).get("_all", 0) / kind_tot
                )

        # Per sample_type breakdown
        for stype, tot in self._eval_acc.get("v12_kind_total", {}).items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/v12_kind_match_{stype}"] = (
                self._eval_acc["v12_kind_match"].get(stype, 0) / tot
            )
            out[f"eval/v12_format_valid_{stype}"] = (
                self._eval_acc["v12_format_valid"].get(stype, 0) / tot
            )
            # Per-stype: emit rate of each observed kind. Reading guide:
            #   stype=silent + v12_answer_emit_rate_silent → should be ~1
            #   stype=silent + v12_recall_emit_rate_silent → should be ~0
            #     (if >0 the model wrongly recalls during silent chunks)
            for obs_band, name in [
                ("v12_observed_answer", "answer_emit_rate"),
                ("v12_observed_recall", "recall_emit_rate"),
                ("v12_observed_compress", "compress_emit_rate"),
            ]:
                out[f"eval/v12_{name}_{stype}"] = (
                    self._eval_acc.get(obs_band, {}).get(stype, 0) / tot
                )
            # Silent samples: also surface "did the answer come out empty?"
            if stype == "silent":
                out[f"eval/v12_silent_empty_rate"] = (
                    self._eval_acc.get("v12_silent_empty_match", {}).get(stype, 0)
                    / tot
                )

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

    # -----------------------------------------------------------------
    # Train-time per-class metrics (flushed to wandb every logging_steps)
    # Tracks per-action-type loss/sample-weight + teacher-forced action
    # argmax accuracy so we can see live whether class balancing is
    # working AND whether the model is learning the action keyword per
    # class (catches compress collapse early).
    # -----------------------------------------------------------------

    def _reset_train_metrics(self):
        self._train_metrics = {
            "loss_sum":   defaultdict(float),
            "loss_n":     defaultdict(int),
            "weight_sum": defaultdict(float),
        }

    def _accumulate_train_metrics(
        self, *, per_sample_loss, sample_weights, sample_meta,
        eval_meta, logits, input_ids,
    ) -> None:
        if not sample_meta:
            return
        psl = (
            per_sample_loss.float().detach().cpu().tolist()
            if per_sample_loss is not None else None
        )
        sw = None
        if sample_weights is not None and sample_weights.numel() > 0:
            sw = sample_weights.float().detach().cpu().tolist()

        # Per-class loss + weight + count
        for i, meta in enumerate(sample_meta):
            stype = (meta.get("sample_type") or "?")
            if psl is not None and i < len(psl):
                self._train_metrics["loss_sum"][stype] += psl[i]
                self._train_metrics["loss_sum"]["_all"] += psl[i]
            self._train_metrics["loss_n"][stype] += 1
            self._train_metrics["loss_n"]["_all"] += 1
            w = sw[i] if sw is not None and i < len(sw) else 1.0
            self._train_metrics["weight_sum"][stype] += w
            self._train_metrics["weight_sum"]["_all"] += w

        # v12: action_keyword_positions doesn't exist (no <action> vocab).
        # Per-class loss + weight is the only signal we accumulate here.

    def _flush_train_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n_total = self._train_metrics["loss_n"].get("_all", 0)
        if n_total == 0:
            return out
        # Per-class loss + weight + sample fraction
        for stype, n in self._train_metrics["loss_n"].items():
            if n == 0:
                continue
            suffix = "" if stype == "_all" else f"_{stype}"
            out[f"train/loss_by_class{suffix}"] = (
                self._train_metrics["loss_sum"][stype] / n
            )
            out[f"train/sw_mean{suffix}"] = (
                self._train_metrics["weight_sum"][stype] / n
            )
            if stype != "_all":
                out[f"train/n_frac_{stype}"] = n / n_total
        self._reset_train_metrics()
        return out

    def log(self, logs, *args, **kwargs):
        """Inject per-class train metrics whenever HF Trainer logs in train mode.

        log() fires from _maybe_log_save_evaluate at every logging_steps and
        also from evaluate() with eval_loss. We only flush train accumulators
        when self.model is actually training; otherwise the eval-loss log
        would prematurely consume a partial bucket.
        """
        if self.model is not None and self.model.training:
            extra = self._flush_train_metrics()
            if extra:
                logs = {**logs, **extra}
        return super().log(logs, *args, **kwargs)


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
