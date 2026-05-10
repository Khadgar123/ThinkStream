"""Training arguments for per-timestep agent SFT.

Based on Qwen3-VL official finetune arguments, extended with:
- Agent protocol special tokens
- LLaMA-Factory ShareGPT messages ingestion
- DeepEyes-style assistant-span loss masking
"""

import transformers
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    # v12 protocol REQUIRES Qwen3-VL (Qwen2.5-VL's chat_template has no
    # tools support). Pass --model_name_or_path Qwen/Qwen3-VL-8B-Instruct.
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen3-VL-8B-Instruct")
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
    eval_balance_strategy: str = field(
        default="none",
        metadata={
            "help": "Eval subsampling strategy when eval_max_samples is set. "
            "'none' keeps the historical Random(0) sample. "
            "'loss_class' balances eval rows across loss_class buckets. "
            "'loss_class_silent_diverse' also stratifies silent rows using "
            "the pass5-style pending/post-answer/no-question and temporal "
            "boundary buckets."
        },
    )
    eval_balance_target_ratios: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional comma-separated eval class target ratios, e.g. "
            "'silent=0.2,response=0.2,recall=0.2,post_recall=0.2,compress=0.2'. "
            "If unset, present loss_class buckets are sampled equally."
        },
    )
    eval_balance_seed: int = field(
        default=0,
        metadata={"help": "Deterministic seed for balanced eval subsampling."},
    )
    model_type: str = field(default="qwen2.5vl")

    # Image
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    # Video (per-timestep: 24 frames, fixed resolution)
    video_max_frames: Optional[int] = field(default=32)
    video_min_frames: Optional[int] = field(default=4)
    # v12.12 (2026-05-02): RUNTIME profile (was 100352/150528). Empirically
    # measured min=130k max=220k → ~235 tok/frame at 32-frame visual window
    # = 7,520 tok in 16K context. Same values as RUNTIME_MM_PROCESSOR_KWARGS
    # in scripts/agent_data_v5/config.py — pass2/SFT/RL/Eval/deploy unified.
    video_max_pixels: int = field(default=220_000)   # ~470x470 area
    video_min_pixels: int = field(default=130_000)   # ~360x360 area
    video_fps: float = field(default=2.0)

    # Per-timestep agent config
    # v12.5: 1s/chunk, 16-chunk visual window (16s @ 2fps = 32 frames).
    agent_chunk_sec: float = field(default=1.0)
    visual_window_chunks: int = field(default=16)
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
    include_failed_verification: bool = field(
        default=False,
        metadata={
            "help": "Include samples with verification.passed=False. "
            "Default is False for the main cold-start SFT path so the model "
            "learns from clean teacher-forced rows. Set True only for "
            "robustness/continuity ablations that intentionally keep tagged "
            "verification failures."
        },
    )
    class_loss_target_ratios: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional comma-separated sample_type target distribution, "
            "e.g. 'silent=0.35,response=0.25,recall=0.25,compress=0.15'. "
            "When set, rows remain unique and per-sample weights are assigned "
            "as w=(target/observed)^alpha normalized to mean 1."
        },
    )
    class_loss_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Strength for class_loss_target_ratios. 1.0 fully matches "
            "the target effective distribution; 0.5 is a softer correction; "
            "0 disables class weighting."
        },
    )
    class_loss_max_weight: float = field(
        default=8.0,
        metadata={
            "help": "Clamp per-class sample weights to this maximum before "
            "renormalizing. Set <=0 to disable clamping."
        },
    )
    compress_token_weighting: bool = field(
        default=True,
        metadata={
            "help": "For compress SFT rows, redistribute loss inside the "
            "assistant span: emphasize tool/action/schema/closing tokens and "
            "down-weight the open-ended summary body. This targets the hard "
            "format/stop behaviour without teaching the model to simply keep "
            "writing longer summaries."
        },
    )
    compress_structure_token_weight: float = field(
        default=2.0,
        metadata={
            "help": "Per-token loss weight for compress structure tokens "
            "outside the summary body, including think/action text, tool tags, "
            "JSON keys, braces, and time_range."
        },
    )
    compress_body_token_weight: float = field(
        default=0.35,
        metadata={
            "help": "Per-token loss weight for the free-form compress summary "
            "text body."
        },
    )
    compress_close_token_weight: float = field(
        default=4.0,
        metadata={
            "help": "Per-token loss weight for compress closing tokens after "
            "the summary body, including the closing quote/braces, tool tag, "
            "and chat end marker."
        },
    )
    compress_close_tail_tokens: int = field(
        default=24,
        metadata={
            "help": "Fallback number of tail tokens to emphasize as close "
            "tokens if exact summary-body token alignment cannot be resolved."
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
