"""
Baseline model evaluation utilities.

Two modes corresponding to two rows in the results table:

1. **Offline baseline** (batch mode):
   Loads N frames uniformly sampled from full video, standard model.generate.
   Matches "Open-source Offline Models" in paper Table 2.
   Uses vanilla HF model (no streaming patches).

2. **Online baseline** (streaming mode):
   Same streaming engine as ThinkStream (chunk-by-chunk, same visual window,
   same KV cache) but WITHOUT think/action protocol.
   Matches "Open-source Online Models" in paper Table 2.
   Uses MODEL_CLS with streaming patches, baseline_sample_restricted.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

# Offline baseline uses standard HF model classes (no streaming patches)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

BASELINE_MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl": Qwen3VLForConditionalGeneration,
}

_EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from eval_common import (
    NoPadDistributedSampler,
    add_common_args,
    build_results,
    load_model_and_processor,
    save_results,
    setup_distributed,
    cleanup_distributed,
)


# ---------------------------------------------------------------------------
# Streaming baseline: sample function (no think/action protocol)
# ---------------------------------------------------------------------------


def baseline_sample_restricted(
    next_token: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    generated_tokens: torch.Tensor,
    generated_length: torch.Tensor,
    restricted_token_ids: list,
    eos_token_id: int,
    **kwargs,
) -> torch.Tensor:
    """Baseline restricted sampling: directly pick the best option token.

    No think budget, no <think>/<response>/<silent> protocol.
    Step 0: pick argmax over restricted_token_ids from logits.
    Step 1+: force EOS.

    Same interface as think_budget_sample_restricted for compatibility with
    streaming_video_chat's sample parameter.
    """
    device = next_token.device

    if step == 0:
        restricted_ids = torch.tensor(
            restricted_token_ids, device=device, dtype=torch.long
        )
        restricted_logits = logits[:, restricted_ids]  # [B, R]
        top1_local = restricted_logits.argmax(dim=-1)  # [B]
        top1_token = restricted_ids[top1_local]  # [B]
        return top1_token.unsqueeze(1)
    else:
        return torch.full_like(next_token, eos_token_id)


# ---------------------------------------------------------------------------
# Offline baseline: load model + video frames + generate
# ---------------------------------------------------------------------------


def load_offline_model(
    model_path: str,
    local_rank: int = 0,
    model_type: str = "qwen3vl",
    min_pixels: int = 200704,
    max_pixels: int = 401408,
):
    """Load a vanilla HF VLM (no streaming patches) for offline evaluation."""
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


def _load_video_frames(
    video_path: str,
    video_start: float = 0.0,
    video_end: float = None,
    max_frames: int = 64,
):
    """Load video frames as PIL Images, uniformly sampled from [start, end]."""
    from decord import VideoReader, cpu
    from PIL import Image

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
    return [Image.fromarray(f) for f in frames]


class OfflineMCQDataset(Dataset):
    """JSONL dataset for offline baseline evaluation."""

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


def add_offline_args(parser):
    """Add offline baseline CLI arguments."""
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, default="qwen3vl",
        choices=list(BASELINE_MODEL_CLS.keys()),
    )
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--max_frames", type=int, default=64,
                        help="Frames uniformly sampled from full video (paper uses 64).")
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--sample", type=int, default=None)
    return parser


@torch.inference_mode()
def offline_predict_mcq(
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
    sample: int = None,
):
    """Offline MCQ prediction: N frames uniformly sampled, standard generate.

    Matches "Open-source Offline Models" row in paper Table 2.
    """
    dataset = OfflineMCQDataset(benchmark_path, sample=sample)

    if world_size > 1:
        sampler = NoPadDistributedSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        collate_fn=lambda batch: batch[0],
    )

    predictions = []
    datums = []
    local_indices = []

    for idx, datum in tqdm.tqdm(
        dataloader, desc="Offline eval", disable=(rank != 0)
    ):
        try:
            video_path = os.path.join(dataset.data_dir, datum["video"])
            video_end = datum.get("video_end")
            video_start = datum.get("video_start", 0.0)

            if "options" in datum and datum["options"]:
                query = (
                    question_prefix + datum["question"] + "\n"
                    + "\n".join(datum["options"]) + question_postfix
                )
            else:
                query = datum["question"]

            frames = _load_video_frames(
                video_path, video_start=video_start,
                video_end=video_end, max_frames=max_frames,
            )
            if not frames:
                raise ValueError(f"No frames loaded from {video_path}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=None, videos=[frames],
                padding=True, return_tensors="pt",
            ).to(model.device)

            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

            generated = output_ids[0, inputs["input_ids"].shape[1]:]
            generated_text = processor.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()

            pred_idx = 0
            gen_upper = generated_text.upper().strip()
            for i, opt in enumerate(options):
                if gen_upper.startswith(opt.upper()):
                    pred_idx = i
                    break

            if rank == 0:
                print(f"[Offline] {generated_text} -> {options[pred_idx]}")

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
