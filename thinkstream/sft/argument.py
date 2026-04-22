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
        default=8192,
        metadata={"help": "Filter samples exceeding this token count in Dataset init (P0-4)."},
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
