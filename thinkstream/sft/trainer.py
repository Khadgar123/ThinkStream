"""Per-timestep agent SFT trainer.

Only keeps what we actually use:
- WeightedSFTTrainer: assistant-span CE + audit metrics
- create_optimizer: per-component learning rate (vision_tower_lr, mm_projector_lr)
- print_trainable_parameters: debugging utility

Removed from Qwen3-VL official finetune:
- flash_attention_forward / varlen attention (data_flatten=False, not needed)
- qwen2vl_forward (Qwen2-VL not supported)
- replace_qwen2_vl_attention_class (only for data_flatten mode)
"""

import json
import os
import time

import torch
import torch.distributed as dist
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from transformers import Trainer
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
    get_length_grouped_indices,
)
from torch.utils.data import Sampler
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


def _split_to_even_chunks(
    indices: List[int],
    chunk_size: int,
    *,
    pad_pool: List[int],
    seed: int,
) -> List[List[int]]:
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]
    if chunks and len(chunks[-1]) < chunk_size and pad_pool:
        generator = torch.Generator()
        generator.manual_seed(seed)
        need = chunk_size - len(chunks[-1])
        fill = torch.randint(0, len(pad_pool), (need,), generator=generator).tolist()
        chunks[-1] = chunks[-1] + [pad_pool[i] for i in fill]
    return chunks


def _modality_grouped_indices(
    modality_lengths: List[int],
    batch_size: int,
    seed: int = 20260506,
) -> List[int]:
    """Group visual and text-only rows into separate global batches.

    ZeRO3 can hang when different ranks execute different module paths in the
    same collective window. Older datasets may contain text-only inter-chunk
    rows while normal rows execute the vision path. Keeping a global batch
    single-modality makes all ranks agree on whether vision/merger parameters
    are touched.
    """
    visual = [i for i, l in enumerate(modality_lengths) if l >= 0]
    text = [i for i, l in enumerate(modality_lengths) if l < 0]

    def _ordered(pool: List[int], salt: int) -> List[int]:
        if not pool:
            return []
        lengths = [abs(int(modality_lengths[i])) for i in pool]
        generator = torch.Generator()
        generator.manual_seed(seed + salt)
        local_order = get_length_grouped_indices(
            lengths, batch_size, generator=generator,
        )
        return [pool[i] for i in local_order]

    chunks = (
        _split_to_even_chunks(
            _ordered(visual, 17), batch_size,
            pad_pool=visual, seed=seed + 101,
        )
        + _split_to_even_chunks(
            _ordered(text, 31), batch_size,
            pad_pool=text, seed=seed + 151,
        )
    )
    if not chunks:
        return []

    # Shuffle batch order while preserving single-modality composition inside
    # each chunk. Use torch RNG so Trainer/Accelerate seeding still controls it.
    generator = torch.Generator()
    generator.manual_seed(seed + 47)
    perm = torch.randperm(len(chunks), generator=generator).tolist()
    return [idx for p in perm for idx in chunks[p]]


class ModalityGroupedSampler(Sampler):
    def __init__(self, batch_size: int, modality_lengths: List[int], seed: int = 20260506):
        self.batch_size = int(batch_size)
        self.modality_lengths = list(modality_lengths)
        self.seed = int(seed)
        self._indices = _modality_grouped_indices(
            self.modality_lengths, self.batch_size, seed=self.seed,
        )

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        return iter(self._indices)


def expected_v12_kind_for_eval(
    stype: str,
    *,
    turn_idx: int = 0,
    n_turns: int = 1,
    action: str = "",
) -> str:
    """Expected assistant behavior for teacher-forced protocol metrics."""
    if stype in ("recall_query", "recall") and n_turns >= 2:
        if turn_idx == 0:
            return "recall"
        return (
            "answer_empty"
            if (action or "").strip().lower() == "silent"
            else "answer_nonempty"
        )
    if stype == "silent":
        return "answer_empty"
    if stype in ("response", "recall_response"):
        return "answer_nonempty"
    if stype in ("recall_query", "recall"):
        return "recall"
    if stype in ("compress", "compress_inter"):
        return "compress"
    return "unknown"


# ---------------------------------------------------------------------------
# Assistant-span SFT Trainer
# ---------------------------------------------------------------------------

class WeightedSFTTrainer(Trainer):
    """HF Trainer subclass with audit logging + per-class eval metrics.

    Assistant-span CE with optional per-sample class reweighting.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audit_step = self._init_audit_writers()
        self._reset_eval_accumulator()
        self._reset_train_metrics()

    def _get_train_sampler(self, train_dataset=None):
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None:
            return None

        if (
            os.environ.get("THINKSTREAM_GROUP_BY_MODALITY", "1") == "1"
            and hasattr(train_dataset, "modality_lengths")
        ):
            world_size = max(1, int(getattr(self.args, "world_size", 1) or 1))
            per_rank_bsz = int(getattr(self.args, "per_device_train_batch_size", 1) or 1)
            grad_accum = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
            # Group by the full data-parallel window. Accelerate/Trainer
            # shards this global order across ranks; if the group is only a
            # per-rank batch, different ranks can still see different
            # modalities at the same optimizer step.
            batch_size = per_rank_bsz * world_size * grad_accum
            seed = int(os.environ.get("THINKSTREAM_SFT_SAMPLER_SEED", "20260506"))
            return ModalityGroupedSampler(
                batch_size, train_dataset.modality_lengths, seed=seed,
            )

        if self.args.group_by_length and hasattr(train_dataset, "lengths"):
            batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
            return LengthGroupedSampler(batch_size, lengths=train_dataset.lengths)

        return super()._get_train_sampler(train_dataset)

    def _get_eval_sampler(self, eval_dataset):
        """Keep eval global batches single-modality under ZeRO3.

        Training already groups rows by whether they execute the vision path.
        Eval must do the same; otherwise one rank can all-gather vision
        parameters while another rank never touches the vision tower, which
        trips NCCL/ZeRO3 collectives.
        """
        if eval_dataset is None:
            return None

        group_eval = os.environ.get(
            "THINKSTREAM_GROUP_EVAL_BY_MODALITY",
            os.environ.get("THINKSTREAM_GROUP_BY_MODALITY", "1"),
        )
        if group_eval == "1" and hasattr(eval_dataset, "modality_lengths"):
            world_size = max(1, int(getattr(self.args, "world_size", 1) or 1))
            per_rank_bsz = int(getattr(self.args, "per_device_eval_batch_size", 1) or 1)
            # Eval has no gradient accumulation window. Group by the full
            # data-parallel batch that Accelerate later shards across ranks.
            batch_size = per_rank_bsz * world_size
            seed = int(os.environ.get(
                "THINKSTREAM_SFT_EVAL_SAMPLER_SEED",
                os.environ.get("THINKSTREAM_SFT_SAMPLER_SEED", "20260506"),
            ))
            return ModalityGroupedSampler(
                batch_size, eval_dataset.modality_lengths, seed=seed + 1009,
            )

        return super()._get_eval_sampler(eval_dataset)

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
        sample_weights = inputs.pop("sample_weights", None)
        token_loss_weight = inputs.pop("token_loss_weight", None)
        sample_meta = inputs.pop("sample_meta", None)
        eval_meta = inputs.pop("eval_meta", None)
        # Keep input_ids handy for argmax-vs-gold accumulation during eval
        eval_input_ids = inputs["input_ids"] if not self.model.training else None

        debug_payload = None
        debug_enabled = (
            self.model.training
            and os.environ.get("THINKSTREAM_SFT_DEBUG_TIMING", "0") == "1"
        )
        if debug_enabled:
            debug_payload = self._debug_forward_payload(inputs, sample_meta)
            self._write_debug_event("before_forward", debug_payload)
            if (
                os.environ.get("THINKSTREAM_SFT_DEBUG_SYNC", "0") == "1"
                and torch.cuda.is_available()
            ):
                self._write_debug_event("before_pre_forward_sync", debug_payload)
                torch.cuda.synchronize()
                self._write_debug_event("after_pre_forward_sync", debug_payload)
            t_forward = time.time()

        outputs = model(**inputs)

        if debug_enabled:
            assert debug_payload is not None
            debug_payload["forward_sec"] = round(time.time() - t_forward, 4)
            self._write_debug_event("after_forward", debug_payload)
            if (
                os.environ.get("THINKSTREAM_SFT_DEBUG_SYNC", "0") == "1"
                and torch.cuda.is_available()
            ):
                self._write_debug_event("before_post_forward_sync", debug_payload)
                torch.cuda.synchronize()
                self._write_debug_event("after_post_forward_sync", debug_payload)

        loss = outputs.loss
        per_sample_loss_for_audit = None

        if self.model.training and inputs.get("labels") is not None:
            per_sample_loss_for_audit = self._per_sample_ce_loss(
                outputs.logits,
                inputs["labels"],
                token_loss_weight=token_loss_weight,
            )
            if sample_weights is not None and sample_weights.numel() > 0:
                weights = sample_weights.to(
                    device=per_sample_loss_for_audit.device,
                    dtype=per_sample_loss_for_audit.dtype,
                ).view(-1)
                if weights.numel() == per_sample_loss_for_audit.numel():
                    loss = (
                        per_sample_loss_for_audit * weights
                    ).sum() / weights.sum().clamp_min(1e-6)

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

    def _debug_forward_payload(self, inputs, sample_meta):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        def _shape(name):
            x = inputs.get(name)
            return list(x.shape) if torch.is_tensor(x) else None

        return {
            "rank": int(rank),
            "global_step": int(getattr(self.state, "global_step", -1)),
            "audit_step_next": None if self._audit_step is None else int(self._audit_step + 1),
            "input_ids_shape": _shape("input_ids"),
            "labels_shape": _shape("labels"),
            "pixel_values_videos_shape": _shape("pixel_values_videos"),
            "video_grid_thw_shape": _shape("video_grid_thw"),
            "pixel_values_shape": _shape("pixel_values"),
            "image_grid_thw_shape": _shape("image_grid_thw"),
            "sample_meta": sample_meta or [],
        }

    def _write_debug_event(self, event, payload):
        try:
            rank = payload.get("rank", 0)
            out_dir = Path(self.args.output_dir) / "audit"
            out_dir.mkdir(parents=True, exist_ok=True)
            row = dict(payload)
            row["event"] = event
            row["ts"] = time.time()
            with (out_dir / f"sft_debug_rank{rank}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _per_sample_ce_loss(self, logits, labels, token_loss_weight=None):
        """Mean assistant-token CE per sample.

        Class weights are applied after this average so long assistant
        responses do not dominate merely because they have more tokens.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid = shift_labels.ne(IGNORE_INDEX)
        if not valid.any():
            return torch.zeros(
                labels.size(0),
                device=logits.device,
                dtype=logits.dtype,
            )

        valid_index = valid.nonzero(as_tuple=False)
        sample_index = valid_index[:, 0]
        flat_logits = shift_logits[valid]
        flat_labels = shift_labels[valid]
        flat_loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
        )
        if token_loss_weight is not None:
            tw = token_loss_weight[..., 1:].to(
                device=flat_loss.device,
                dtype=flat_loss.dtype,
            )
            flat_weight = tw[valid]
            flat_loss = flat_loss * flat_weight
        else:
            flat_weight = torch.ones_like(flat_loss)

        batch = labels.size(0)
        loss_sum = torch.zeros(batch, device=flat_loss.device, dtype=flat_loss.dtype)
        denom = torch.zeros(batch, device=flat_loss.device, dtype=flat_loss.dtype)
        loss_sum = loss_sum.scatter_add(0, sample_index, flat_loss)
        denom = denom.scatter_add(0, sample_index, flat_weight)
        return loss_sum / denom.clamp_min(1.0)

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
            psl = per_sample_loss.detach().float().cpu()
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
                # v12.11 audit fix #4 (2026-05-01): iterate ALL assistant
                # spans, not just the first. Multi-turn recall samples have
                # 2 assistant turns (tool_call + final answer); the
                # legacy single-span loop missed the final-answer turn
                # in eval metrics even though loss did train it.
                # data_processor.py:eval_meta now exposes "ans_spans" (list
                # of (start, end) tuples). Fall back to single-span tuple
                # for backward compat with older cached eval batches.
                ans_spans = meta.get("ans_spans")
                if not ans_spans:
                    ans_start = meta.get("ans_start")
                    ans_end = meta.get("ans_end")
                    ans_spans = (
                        [(ans_start, ans_end)]
                        if ans_start is not None and ans_end is not None
                        else []
                    )
                for turn_idx, (ans_start, ans_end) in enumerate(ans_spans):
                    if ans_start is None or ans_end is None:
                        continue
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
                    # sample_type. v12.11: now invoked per assistant turn,
                    # so multi-turn recall samples get BOTH tool_call and
                    # final-answer turns counted toward behavioral stats.
                    # turn_idx tells the helper which turn this span is so
                    # it can pick the right expected_kind for shape-B
                    # recall (turn 0 = tool_call, turn 1 = final answer).
                    self._accumulate_v12_behavioral(
                        preds, input_ids, b, s, e, stype,
                        turn_idx=turn_idx,
                        n_turns=len(ans_spans),
                        action=meta.get("action") or meta.get("gold_action", ""),
                    )

    def _accumulate_v12_behavioral(
        self, preds, input_ids, b: int, s: int, e: int, stype: str,
        turn_idx: int = 0, n_turns: int = 1,
        action: str = "",
    ) -> None:
        """v12.1 per-sample behavioral counters from teacher-forced argmax.

        v12.11 audit-3 fix (2026-05-01): turn_idx + n_turns added so this
        helper can disambiguate the EXPECTED kind for multi-turn recall
        samples (shape B = [tool_call, final_answer]). For sample_type
        "recall" with n_turns=2, turn 0 expects "recall"; turn 1 expects a
        non-empty answer for recall+response and an empty answer for
        recall-failure→silent. Single-turn samples keep the prior behavior.
        """
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

        # v12.11 audit-3 fix: shape-B recall samples have 2 assistant turns;
        # turn 0 should be tool_call ("recall"), turn 1 should be final
        # answer ("answer_nonempty"). Inferring expected purely from
        # sample_type would tag turn 1 as "recall" too → false negative on
        # v12_kind_match.
        expected_kind = expected_v12_kind_for_eval(
            stype, turn_idx=turn_idx, n_turns=n_turns, action=action,
        )

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
            "loss_value_n": defaultdict(int),
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
                self._train_metrics["loss_value_n"][stype] += 1
                self._train_metrics["loss_value_n"]["_all"] += 1
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
            loss_n = self._train_metrics["loss_value_n"].get(stype, 0)
            if loss_n:
                out[f"train/loss_by_class{suffix}"] = (
                    self._train_metrics["loss_sum"][stype] / loss_n
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
