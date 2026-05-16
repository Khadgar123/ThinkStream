"""Training arguments for trajectory-mixed agent SFT.

Based on Qwen3-VL official finetune arguments, extended with:
- Agent protocol special tokens
- LLaMA-Factory ShareGPT messages ingestion
- Assistant-span loss masking
"""

import transformers
from dataclasses import dataclass, field
from typing import Optional

from thinkstream.data.schema import DEFAULT_VIDEO_MAX_PIXELS, DEFAULT_VIDEO_MIN_PIXELS


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
            "make_trajectory_data_module builds an eval dataset that "
            "the HF Trainer will run on every eval_steps. Use "
            "stream_agent_trajectory_val for the held-out video-disjoint pool."
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
            "'from_start=0.425,from_compress=0.425,compress=0.15'. "
            "If unset, present loss_class buckets are sampled equally."
        },
    )
    eval_balance_seed: int = field(
        default=0,
        metadata={"help": "Deterministic seed for balanced eval subsampling."},
    )
    # Default tracks the model in ModelArguments.model_name_or_path
    # (Qwen3-VL). train.py auto-detects the model and overrides this, but
    # callers that import DataArguments directly (eval, debug scripts) get
    # the right default without that detection. Override explicitly to
    # ``"qwen2.5vl"`` when training the older model.
    model_type: str = field(default="qwen3vl")

    # Image
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)

    # Video chunk settings.
    video_max_frames: Optional[int] = field(default=32)
    video_min_frames: Optional[int] = field(default=4)
    # Streaming-runtime profile selected for the 8B local HF + KV-window path:
    #   min = 256·28·28 = 200,704
    #   max = 512·28·28 = 401,408
    # Same values as DEFAULT_VIDEO_{MIN,MAX}_PIXELS in thinkstream.data.schema
    # and RUNTIME_MM_PROCESSOR_KWARGS in scripts/agent_data/config.py —
    # SFT/RL/Eval/deploy unified.
    video_max_pixels: int = field(default=DEFAULT_VIDEO_MAX_PIXELS)
    video_min_pixels: int = field(default=DEFAULT_VIDEO_MIN_PIXELS)
    video_fps: float = field(default=2.0)

    # Agent trajectory config.
    # v12.15: 1s/chunk, 8-chunk visual window (8s @ 2fps = 16 frames).
    agent_chunk_sec: float = field(default=1.0)
    visual_window_chunks: int = field(default=8)
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
        default=True,
        metadata={
            "help": "Include samples with verification.passed=False. "
            "Default True: pass3e tags but does not delete; SFT keeps the "
            "full trajectory so verifier failures only de-weight via "
            "token_loss_weight (set in collator). Flip to False only when "
            "running a strict cold-start ablation that intentionally drops "
            "verification failures."
        },
    )
    class_loss_target_ratios: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional comma-separated trajectory loss-class target "
            "distribution, e.g. "
            "'from_start=0.425,from_compress=0.425,compress=0.15'. "
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
        default=False,
        metadata={
            "help": "Ablation-only. When enabled for compress SFT rows, "
            "redistribute loss inside the assistant span between format/body/"
            "closing tokens. Default False keeps normal assistant-token CE so "
            "the compact-memory body is trained with the same weight as the format."
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
    action_class_loss_mode: str = field(
        default="none",
        metadata={
            "help": "Action-start loss balancing mode for "
            "{</Silence>, </Response>, optional compress/recall tool-name anchors}. "
            "Response close tags and answer text are ordinary CE targets. "
            "'none' = no extra balancing (legacy behaviour). "
            "'inverse_freq' = inverse-frequency-weighted class-weighted CE applied via "
            "token_loss_weight (compatible with Liger fused CE). "
            "'focal' = focal-modulation focal loss with optional auto-alpha; "
            "replaces the trainer's per-sample CE path (no Liger)."
        },
    )
    action_class_focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": "Focal-loss gamma. 0.0 disables focal modulation. "
            "Only consulted when action_class_loss_mode == 'focal'."
        },
    )
    action_class_focal_auto_alpha: bool = field(
        default=True,
        metadata={
            "help": "When True, focal loss applies per-class inverse-frequency "
            "alpha weights on top of the focal modulation (auto-alpha class "
            "rebalancing). Only consulted when action_class_loss_mode == "
            "'focal'."
        },
    )
    action_class_weight_floor: float = field(
        default=0.05,
        metadata={
            "help": "Lower clamp for per-class weights. Prevents action tokens "
            "from collapsing to zero loss in heavily-imbalanced batches."
        },
    )
    action_class_weight_ceil: float = field(
        default=20.0,
        metadata={
            "help": "Upper clamp for per-class weights. Default "
            "ce_weight clamp(0, 20)."
        },
    )
    loss_bucket_weighting: bool = field(
        default=False,
        metadata={
            "help": "When enabled, replace ordinary assistant-token CE with "
            "two explicit token buckets: action and text. The current "
            "bucketizer maps response/answer/recall toolcall tokens to action "
            "and think/compact-memory tokens to text. The trainer computes "
            "L = lambda_action * mean(CE_action) + lambda_text * mean(CE_text). "
            "Default False preserves the existing CE path."
        },
    )
    loss_bucket_action_weight: float = field(
        default=1.0,
        metadata={"help": "lambda_action for loss_bucket_weighting."},
    )
    loss_bucket_key_weight: float = field(
        default=1.0,
        metadata={"help": "Legacy lambda_key; no-op for the current two-bucket loss."},
    )
    loss_bucket_text_weight: float = field(
        default=1.0,
        metadata={"help": "lambda_text for loss_bucket_weighting."},
    )
    loss_bucket_answer_weight: float = field(
        default=1.0,
        metadata={"help": "Legacy lambda_answer; no-op for the current two-bucket loss."},
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
