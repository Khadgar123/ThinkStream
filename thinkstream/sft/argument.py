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
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=True)
    tune_mm_mlp: bool = field(default=True)
    tune_mm_vision: bool = field(default=False)


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
            "help": "Smoothing exponent on inv-freq weights. "
            "1.0 = pure inverse frequency (most aggressive rebalance), "
            "0.5 = √(inv-freq), 0.0 = uniform. v11.4 (Path B): 1.0 → 0.85. "
            "v11.3's 1.0 over-corrected — silent (70% data) got 6× LESS "
            "sampling than uniform, combined with ACTION_WEIGHTS[compress]=2.5 "
            "produced 7.2× compress/silent gradient ratio and silent_acc "
            "regressed 99%→86% on eval. 0.85 is the calibrated midpoint "
            "(measured ratio ~3.6×, compress still favored but silent no "
            "longer crushed). Don't go back to 0.7 — that's v11.2's value "
            "which caused compress collapse."
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
