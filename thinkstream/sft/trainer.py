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

        # v11.3: optional per-sample uniqueness weighting on top of class-balance.
        extra_weights = None
        if getattr(self.args, "unique_think_weight", False):
            extra_weights = [float(s.get("_unique_rate", 1.0)) for s in ds.samples]

        sampler = ClassBalancedDistributedSampler(
            sample_types=sample_types,
            num_samples=len(sample_types),
            seed=self.args.seed,
            smoothing=getattr(self.args, "class_balance_smoothing", 0.7),
            extra_weights=extra_weights,
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
    # Eval-time argmax accuracy (teacher-forced) — see data_processor's
    # _extract_eval_positions. Tracks per-class:
    #   - action_match / action_total  → eval/action_acc[_<class>]
    #   - post_match   / post_total    → eval/post_action_acc[_<class>]
    # silent_eos_rate is just post_action_acc filtered to silent samples.
    # -----------------------------------------------------------------

    def _reset_eval_accumulator(self):
        self._eval_acc = {
            "action_match":  defaultdict(int),
            "action_total":  defaultdict(int),
            "post_match":    defaultdict(int),
            "post_total":    defaultdict(int),
            # v11.3: per-token argmax counts inside <summary>/<query>/<response>
            # spans. Compress range coverage / recall query format / response
            # content quality, all teacher-forced.
            "summary_match": defaultdict(int),
            "summary_total": defaultdict(int),
            "query_match":   defaultdict(int),
            "query_total":   defaultdict(int),
            # v11.4: response content argmax (response / recall_response samples).
            "response_match": defaultdict(int),
            "response_total": defaultdict(int),
            # v11.4: closed-book format compliance — does the model emit
            # <action> at the pre-action position? action_acc above is
            # teacher-forced "given <action>, what's inside" (open-book);
            # this measures "did you decide to start <action>".
            "format_match":  defaultdict(int),
            "format_total":  defaultdict(int),
        }

    @staticmethod
    def _argmax_match_at(preds, input_ids, b, positions, L) -> tuple:
        """Return (n_match, n_total) for teacher-forced argmax over positions.

        logits[p-1] predicts token at position p, so we compare preds[b, p-1]
        against input_ids[b, p]. Positions outside [1, L) are skipped.
        """
        m = 0; n = 0
        for p in positions:
            if p <= 0 or p >= L:
                continue
            n += 1
            if preds[b, p - 1].item() == int(input_ids[b, p].item()):
                m += 1
        return m, n

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

                # Summary span (compress samples) — token-level argmax acc.
                summary_positions = meta.get("summary_span_positions") or []
                if summary_positions:
                    m, n = self._argmax_match_at(preds, input_ids, b, summary_positions, L)
                    self._eval_acc["summary_total"][stype] += n
                    self._eval_acc["summary_total"]["_all"] += n
                    self._eval_acc["summary_match"][stype] += m
                    self._eval_acc["summary_match"]["_all"] += m

                # Query span (recall_query samples) — token-level argmax acc.
                query_positions = meta.get("query_span_positions") or []
                if query_positions:
                    m, n = self._argmax_match_at(preds, input_ids, b, query_positions, L)
                    self._eval_acc["query_total"][stype] += n
                    self._eval_acc["query_total"]["_all"] += n
                    self._eval_acc["query_match"][stype] += m
                    self._eval_acc["query_match"]["_all"] += m

                # v11.4: response span (response / recall_response samples).
                response_positions = meta.get("response_span_positions") or []
                if response_positions:
                    m, n = self._argmax_match_at(preds, input_ids, b, response_positions, L)
                    self._eval_acc["response_total"][stype] += n
                    self._eval_acc["response_total"]["_all"] += n
                    self._eval_acc["response_match"][stype] += m
                    self._eval_acc["response_match"]["_all"] += m

                # v11.4: closed-book format compliance — at the position
                # before <action>, does the model's argmax equal <action>?
                pre_action_pos = meta.get("pre_action_position")
                action_open_id = meta.get("action_open_token_id")
                if (pre_action_pos is not None and action_open_id is not None
                        and 0 <= pre_action_pos < L):
                    self._eval_acc["format_total"][stype] += 1
                    self._eval_acc["format_total"]["_all"] += 1
                    if preds[b, pre_action_pos].item() == int(action_open_id):
                        self._eval_acc["format_match"][stype] += 1
                        self._eval_acc["format_match"]["_all"] += 1

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
            "action_match", "action_total", "post_match", "post_total",
            "summary_match", "summary_total", "query_match", "query_total",
            # v11.4
            "response_match", "response_total", "format_match", "format_total",
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
        # v11.3: compress range coverage = token-level argmax acc inside
        # <summary>...</summary>. Surfaced overall + per-class so a drop
        # specifically on compress samples is visible.
        sum_tot = self._eval_acc["summary_total"].get("_all", 0)
        if sum_tot > 0:
            out["eval/summary_argmax_acc"] = (
                self._eval_acc["summary_match"].get("_all", 0) / sum_tot
            )
        for stype, tot in self._eval_acc["summary_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/summary_argmax_acc_{stype}"] = (
                self._eval_acc["summary_match"].get(stype, 0) / tot
            )
        # v11.3: recall query format/content quality under teacher forcing.
        q_tot = self._eval_acc["query_total"].get("_all", 0)
        if q_tot > 0:
            out["eval/query_argmax_acc"] = (
                self._eval_acc["query_match"].get("_all", 0) / q_tot
            )
        for stype, tot in self._eval_acc["query_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/query_argmax_acc_{stype}"] = (
                self._eval_acc["query_match"].get(stype, 0) / tot
            )
        # v11.4: response content argmax (response / recall_response samples).
        r_tot = self._eval_acc["response_total"].get("_all", 0)
        if r_tot > 0:
            out["eval/response_argmax_acc"] = (
                self._eval_acc["response_match"].get("_all", 0) / r_tot
            )
        for stype, tot in self._eval_acc["response_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/response_argmax_acc_{stype}"] = (
                self._eval_acc["response_match"].get(stype, 0) / tot
            )
        # v11.4: CLOSED-BOOK format compliance — does the model emit <action>
        # at the pre-action position? Critical contrast against open-book
        # eval/action_accuracy: action_acc=0.98 + format_compliance=0.05
        # is the smoking gun for the cold-start init bug (model knows what
        # token follows <action>, but never emits <action> itself).
        f_tot = self._eval_acc["format_total"].get("_all", 0)
        if f_tot > 0:
            out["eval/format_compliance"] = (
                self._eval_acc["format_match"].get("_all", 0) / f_tot
            )
        for stype, tot in self._eval_acc["format_total"].items():
            if stype == "_all" or tot == 0:
                continue
            out[f"eval/format_compliance_{stype}"] = (
                self._eval_acc["format_match"].get(stype, 0) / tot
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
            "loss_sum":     defaultdict(float),
            "loss_n":       defaultdict(int),
            "weight_sum":   defaultdict(float),
            "action_match": defaultdict(int),
            "action_total": defaultdict(int),
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

        # Teacher-forced action argmax (mirrors eval logic). Skipped if
        # eval_meta absent — that just means the collator didn't emit
        # action_keyword_positions for this batch.
        if eval_meta is None or logits is None or input_ids is None:
            return
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            B, L = preds.shape
            for b, m in enumerate(eval_meta):
                if not m:
                    continue
                stype = (m.get("sample_type") or "?")
                kw_positions = m.get("action_keyword_positions") or []
                if not kw_positions:
                    continue
                self._train_metrics["action_total"][stype] += 1
                self._train_metrics["action_total"]["_all"] += 1
                all_match = True
                for p in kw_positions:
                    if p <= 0 or p >= L:
                        all_match = False
                        break
                    if preds[b, p - 1].item() != int(input_ids[b, p].item()):
                        all_match = False
                        break
                if all_match:
                    self._train_metrics["action_match"][stype] += 1
                    self._train_metrics["action_match"]["_all"] += 1

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
        # Per-class action argmax accuracy
        for stype, tot in self._train_metrics["action_total"].items():
            if tot == 0:
                continue
            suffix = "" if stype == "_all" else f"_{stype}"
            out[f"train/action_argmax_acc{suffix}"] = (
                self._train_metrics["action_match"].get(stype, 0) / tot
            )
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
