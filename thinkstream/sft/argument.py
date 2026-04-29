"""Training arguments for per-timestep agent SFT.

Based on Qwen3-VL official finetune arguments, extended with:
- Agent protocol special tokens
- Per-sample loss weighting
- Phase-based curriculum training
"""

import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    # v12.0: Qwen3-VL is REQUIRED when protocol_version=v12. Qwen2.5-VL's
    # bundled chat_template has NO tools support — passing tools= is a
    # silent no-op (verified against Qwen/Qwen2.5-VL-7B-Instruct
    # chat_template.json). Qwen3-VL renders the full Hermes <tools>...
    # <tool_call>...<tool_response> protocol. See verification report in
    # docs/v12.0_protocol_migration_design.md §11.
    # Default kept as Qwen2.5-VL for v11 backward compat. v12 users must
    # explicitly pass --model_name_or_path Qwen/Qwen3-VL-8B-Instruct.
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_vision: bool = field(default=False)
    add_agent_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "v11.4: industry-convention default is False — agent "
            "structural tags (<action>, <response>, <query>, <summary>, "
            "etc.) are kept as TEXT in assistant content and tokenized as "
            "multi-token sequences by the existing BPE. This sidesteps "
            "the cold-start magnitude bug that v11.3 hit when these tags "
            "were registered as new single-token vocab entries — every "
            "sub-token is already well-trained, so the model learns the "
            "tag sequence naturally without smart_init. DeepEyes / "
            "ReMemR1 / Qwen-VL official finetune all follow this pattern. "
            "Set True only if you specifically need single-token tags for "
            "inference efficiency AND are willing to retrain with smart_init "
            "(see thinkstream/sft/data_processor.py:smart_init_special_token_embeddings). "
            "Caveat: the existing v11.3 ckpt was trained with this flag "
            "implicitly True (legacy register_special_tokens path) and "
            "cannot be directly continued with False — you'd need to "
            "retrain from a fresh base model."
        },
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    eval_dataset_use: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset name(s) for eval_dataset, same syntax as "
            "dataset_use (comma-separated, %% sampling). When set, "
            "make_per_timestep_data_module builds an eval dataset that "
            "the HF Trainer will run on every eval_steps. Use "
            "stream_agent_val for the held-out video-disjoint pool."
        },
    )
    eval_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, randomly subsample the eval set to at most "
            "this many samples (after overlong filtering). Speeds up "
            "in-loop eval when val.jsonl is large; leave unset to use "
            "all val samples."
        },
    )
    model_type: str = field(default="qwen2.5vl")

    # Image
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    # Video (per-timestep: 24 frames, fixed resolution)
    video_max_frames: Optional[int] = field(default=32)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=150528)   # ~388x388
    video_min_pixels: int = field(default=100352)    # ~317x317
    video_fps: float = field(default=1.0)

    # Per-timestep agent config
    agent_chunk_sec: float = field(default=2.0)
    visual_window_chunks: int = field(default=12)
    max_sample_tokens: Optional[int] = field(
        default=12000,
        metadata={
            "help": "Filter samples exceeding this token count in Dataset init "
            "(P0-4). v10 raised 8192→12000 because visual-window samples with "
            "24 frames + memory + queries average ~7900 tokens (p50) and "
            "8192 was filtering 35% of data. With model_max_length=16384, "
            "12000 leaves ~4K margin for collator padding."
        },
    )
    require_pre_extracted_frames: bool = field(
        default=True,
        metadata={
            "help": "Fail loudly if a sample lacks frame_paths. "
            "Online video decoding is ~50× slower than pre-extracted frames; "
            "set False only for one-off smoke tests."
        },
    )

    # Audit / reviewable training logs
    audit_log_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory for per-step JSONL audit logs (loss, weights, "
            "sample ids). Defaults to <output_dir>/audit if unset."
        },
    )
    audit_log_every: int = field(
        default=1,
        metadata={"help": "Write audit log every N steps (1 = every step)."},
    )

    # Class-balanced sampler (single-phase mixed SFT)
    class_balanced_sampler: bool = field(
        default=True,
        metadata={
            "help": "Use ClassBalancedDistributedSampler so rare actions "
            "(recall/compress/response) are not drowned by silent. "
            "Recommended for single-phase mixed SFT. Set False to use the "
            "default HF distributed sampler (uniform)."
        },
    )
    class_balance_smoothing: float = field(
        default=0.85,
        metadata={
            "help": "Smoothing exponent for inverse-frequency class weights. "
            "Effective sampling weight per class ∝ count^(1 - smoothing). "
            "1.0 = pure inv-freq (most aggressive), 0.0 = uniform.\n\n"
            "History: v11.2's 0.7 caused compress collapse. v11.3's 1.0 "
            "over-corrected (silent_acc regressed 99%→86%). v11.4 settled "
            "on 0.85 as calibrated midpoint.\n\n"
            "v12.5 audit (2026-04-29): NEW data distribution after the "
            "MAX_SAMPLES_PER_VIDEO=15 cap removal. With silent=85% (was "
            "70% in v11.4), 0.85 yields effective per-sample-class weights:\n"
            "    silent (15,483)^0.15 = 4.04\n"
            "    compress (2,107)^0.15 = 3.20\n"
            "    response (540)^0.15  = 2.66\n"
            "    recall (99)^0.15     = 1.92\n"
            "  → effective batch: ~34% silent / 27% compress / 22% response / 16% recall\n\n"
            "This is reasonable balance. If you want MORE response signal "
            "(industry survey shows QA-streaming systems range 50-95% silent; "
            "ThinkStream is QA-style so 85% raw is normal):\n"
            "    smoothing=0.95 → ~25% silent / 30% response (very aggressive)\n"
            "    smoothing=0.85 → ~34% silent / 22% response (current default) ✓\n"
            "    smoothing=0.50 → ~55% silent / 14% response (closer to natural)\n"
            "    smoothing=0.00 → 85% silent (raw distribution, no rebalance)\n\n"
            "Don't go below 0.7 unless you've measured silent_acc — historical "
            "compress collapse at 0.7 may be specific to v11 distribution."
        },
    )
    unique_think_weight: bool = field(
        default=False,
        metadata={
            "help": "v11.3: when True, multiply each sample's class-balanced "
            "weight by its memory-uniqueness rate (set/len of recent_thinks). "
            "Down-weights static-scene videos where the teacher correctly "
            "reports 'scene unchanged' but those repeats add no training "
            "signal. ~5% of videos see >5% repetition; the worst case (a "
            "yoga session) is 28% repeated. Off by default — enable when "
            "monitoring train/n_frac_silent shows over-fit on quiet scenes."
        },
    )

    # v11.5: StreamMind-style focal+alpha on action keyword positions.
    # Attacks the "collapse to silent" failure mode that span_weight + class-
    # balanced sampler alone cannot fix: silent samples reach p=0.99 quickly
    # but vanilla CE keeps recording loss>0 there, so its gradient continues
    # pushing silent's logit higher → other action keywords drift to -inf.
    # Focal (1-p)^gamma kills the gradient on already-correct silent tokens;
    # alpha = inv_freq^softening (StreamMind §5.1) lifts rare-class signal
    # without the over-correction of pure inverse frequency.
    focal_alpha_action: bool = field(
        default=True,
        metadata={
            "help": "v11.5: enable focal+alpha multiplier on the action "
            "keyword token position (the token inside <action>...</action>). "
            "Multiplies into the existing token_loss_weight, so span_weight "
            "and class-balanced sampler still apply — focal+alpha just adds "
            "a per-token, per-class scaling on the decision token to fight "
            "silent collapse. Inspired by StreamMind (arXiv 2503.06220 §5.1) "
            "ablation showing focal+alpha >> inverse-freq >> vanilla CE on "
            "stream-decision class imbalance."
        },
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Focal exponent gamma for the (1-p_correct)^gamma factor. "
            "2.0 is RetinaNet/StreamMind default. Higher = more aggressive "
            "down-weighting of confident predictions (pushes harder on hard "
            "examples). Set 0.0 to disable focal but keep alpha."
        },
    )
    alpha_softening: float = field(
        default=0.5,
        metadata={
            "help": "Power applied to inv-freq for the per-class alpha "
            "weight: alpha_c = (1 / P_c)^softening, normalized so silent=1.0. "
            "0.5 = sqrt-inv (StreamMind-style soft); 1.0 = pure inverse "
            "frequency (over-corrects on rare classes); 0.0 = uniform "
            "(disables alpha but keeps focal). 0.5 is the calibrated default "
            "from long-tail literature (CB-Loss / Focal-Loss original)."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=16384)
    mm_projector_lr: Optional[float] = field(default=None)
    vision_tower_lr: Optional[float] = field(default=None)

    # LoRA
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
