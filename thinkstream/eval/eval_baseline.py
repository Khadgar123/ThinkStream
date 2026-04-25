"""
Baseline model evaluation utilities.

Runs vanilla VLMs (e.g., stock Qwen3-VL) on benchmarks WITHOUT streaming
protocol, think budget, or special tokens. Just standard video QA:
load frames → build prompt → model.generate → parse answer.

Used for comparison against ThinkStream streaming evaluation.
"""

import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoProcessor

# Baseline uses standard HF model classes, NOT ThinkStream patched ones
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

BASELINE_MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl": Qwen3VLForConditionalGeneration,
}

# Re-use common utilities that don't depend on streaming
_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_common import (
    NoPadDistributedSampler,
    TeeWriter,
    build_results,
    save_results,
    setup_distributed,
    cleanup_distributed,
)


# ---------------------------------------------------------------------------
# Video loading (non-streaming: all frames at once)
# ---------------------------------------------------------------------------


def _load_video_frames(
    video_path: str,
    video_start: float = 0.0,
    video_end: float = None,
    max_frames: int = 64,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
):
    """Load video frames as a list of PIL Images for standard VLM input.

    Returns frames evenly sampled from [video_start, video_end].
    """
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu())
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = int(video_start * fps)
    end_frame = int(video_end * fps) if video_end else total_frames
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_frames))

    n_available = end_frame - start_frame
    n_sample = min(max_frames, n_available)

    if n_sample <= 0:
        return []

    indices = np.linspace(start_frame, end_frame - 1, n_sample, dtype=int)
    frames = vr.get_batch(indices.tolist()).asnumpy()

    from PIL import Image
    return [Image.fromarray(f) for f in frames]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_baseline_model(
    model_path: str,
    local_rank: int = 0,
    model_type: str = "qwen3vl",
    min_pixels: int = 200704,
    max_pixels: int = 401408,
):
    """Load a vanilla HF VLM (no ThinkStream patches)."""
    if model_type not in BASELINE_MODEL_CLS:
        raise ValueError(
            f"Unsupported model_type: {model_type}. "
            f"Choose from {list(BASELINE_MODEL_CLS.keys())}"
        )

    model = BASELINE_MODEL_CLS[model_type].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=f"cuda:{local_rank}",
    )

    processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
    vp = processor.video_processor
    vp.max_pixels = max_pixels
    vp.min_pixels = min_pixels
    vp.size["shortest_edge"] = min_pixels
    vp.size["longest_edge"] = max_pixels

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def add_baseline_args(parser):
    """Add baseline evaluation CLI arguments."""
    parser.add_argument(
        "--benchmark_dir", type=str, required=True, help="Path to benchmark directory."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model directory."
    )
    parser.add_argument(
        "--model_type", type=str, default="qwen3vl",
        choices=list(BASELINE_MODEL_CLS.keys()),
    )
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=64,
                        help="Max frames to sample from video.")
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N items for debugging.")
    return parser


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BaselineMCQDataset(Dataset):
    """JSONL dataset for baseline evaluation. No streaming, no preloading."""

    def __init__(self, path, sample=None):
        lines = open(path).readlines()
        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)
        self.datums = [json.loads(line) for line in lines]
        self.data_dir = os.path.dirname(path)

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        return i, self.datums[i]


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------


@torch.inference_mode()
def baseline_predict_mcq(
    model,
    processor,
    benchmark_path: str,
    options: list,
    question_prefix: str = "",
    question_postfix: str = "\nPlease select the correct answer.",
    max_new_tokens: int = 30,
    max_frames: int = 64,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    rank: int = 0,
    world_size: int = 1,
    model_type: str = "qwen3vl",
    sample: int = None,
):
    """Standard VLM MCQ prediction — no streaming, no think budget.

    For each question: load all frames up to video_end, build a standard
    prompt, run model.generate, parse the answer token.
    """
    dataset = BaselineMCQDataset(benchmark_path, sample=sample)

    if world_size > 1:
        sampler = NoPadDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=lambda batch: batch[0],
    )

    predictions = []
    datums = []
    local_indices = []

    for idx, datum in tqdm.tqdm(
        dataloader, desc="Baseline eval", disable=(rank != 0)
    ):
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])
            video_end = datum.get("video_end")
            video_start = datum.get("video_start", 0.0)

            # Build question with options
            if "options" in datum and datum["options"]:
                query = (
                    question_prefix
                    + datum["question"]
                    + "\n"
                    + "\n".join(datum["options"])
                    + question_postfix
                )
            else:
                query = datum["question"]

            # Load frames
            frames = _load_video_frames(
                video_path,
                video_start=video_start,
                video_end=video_end,
                max_frames=max_frames,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

            if not frames:
                raise ValueError(f"No frames loaded from {video_path}")

            # Build messages in standard Qwen VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            # Process with HF processor
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=None,
                videos=[frames],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # Standard generation
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            # Extract generated tokens (skip input)
            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            generated_text = processor.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()

            # Parse answer: find first matching option
            pred_idx = 0
            gen_upper = generated_text.upper().strip()
            for i, opt in enumerate(options):
                if gen_upper.startswith(opt.upper()):
                    pred_idx = i
                    break

            if rank == 0:
                print(f"[Baseline] {generated_text} -> {options[pred_idx]}")

            predictions.append(pred_idx)
            datums.append({**datum, "success": True})
            local_indices.append(idx)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            predictions.append(random.randint(0, len(options) - 1))
            datums.append({**datum, "success": False})
            local_indices.append(idx)
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Gather results across ranks
    if world_size > 1:
        local_data = list(zip(local_indices, predictions, datums))
        gathered_data = [None] * world_size
        dist.all_gather_object(gathered_data, local_data)
        if rank == 0:
            all_data = []
            for proc_data in gathered_data:
                all_data.extend(proc_data)
            all_data.sort(key=lambda x: x[0])
            return np.array([d[1] for d in all_data]), [d[2] for d in all_data], 0
        else:
            return np.array(predictions), datums, rank
    else:
        return np.array(predictions), datums, 0
