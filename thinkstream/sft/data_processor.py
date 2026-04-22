"""Per-timestep agent SFT data processor.

Based on Qwen3-VL official finetune data_processor.py, adapted for
per-timestep independent samples. Each sample = one inference step snapshot.

Key differences from standard VLM SFT:
- Input is structured pipeline JSON (not conversations format)
- Messages contain <memory>, <visual_window>, <recalled_frames> tags
- Per-sample loss weight by action type
- Single assistant turn per sample (per-timestep design)

See docs/sft_engineering.md §2 and docs/data_construction_zh.md §13.
"""

import json
import random
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from .data_list import data_list
from .rope2d import get_rope_index_25, get_rope_index_3

IGNORE_INDEX = -100

# Per-sample loss weights by action type (sft_engineering.md §5.2)
ACTION_WEIGHTS = {
    "silent": 0.5,
    "response": 1.0,
    "recall_query": 1.5,
    "recall_response": 1.0,
    "compress": 1.5,
    "merge_compress": 1.5,
}

# Agent special tokens (data_construction_zh.md §13.2, Approach B)
SPECIAL_TOKENS_AGENT = [
    # Action protocol
    "<think>", "</think>",
    "<action>", "</action>",
    "<silent>", "<response>", "</response>",
    "<query>", "</query>",
    "<recall_result>", "</recall_result>",
    # Input structure tags
    "<memory>", "</memory>",
    "<compressed>", "</compressed>",
    "<pending>", "</pending>",
    "<visual_window>", "</visual_window>",
    "<recalled_frames>", "</recalled_frames>",
    "<user_input>", "</user_input>",
    # Output payload
    "<summary>", "</summary>",
    # Trigger
    "<compress_trigger>", "</compress_trigger>",
]

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# Processor configuration
# ---------------------------------------------------------------------------

def update_processor_pixels(processor, data_args):
    """Configure image/video processor resolution limits.

    Must update BOTH pixel attrs AND size dict — some processor
    implementations check size dict for resize decisions.
    """
    ip = processor.image_processor
    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels

    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps

    return processor


def register_special_tokens(processor, model_type: str):
    """Add agent protocol special tokens to tokenizer.

    Qwen3-VL natively supports <think>/<∕think>, so skip those for qwen3vl.
    """
    tokenizer = processor.tokenizer

    if model_type == "qwen3vl":
        tokens_to_add = [t for t in SPECIAL_TOKENS_AGENT
                         if t not in ("<think>", "</think>")]
    else:
        tokens_to_add = list(SPECIAL_TOKENS_AGENT)

    # Only add tokens that don't already exist
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in tokens_to_add if t not in existing]

    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
        rank0_print(f"Added {num_added} special tokens: {new_tokens}")

    return processor


# ---------------------------------------------------------------------------
# Message construction (pipeline JSON → Qwen chat messages)
# ---------------------------------------------------------------------------

# Import shared protocol for memory formatting.
# The canonical format_memory_block lives in agent_protocol to guarantee
# train/inference identity. This wrapper handles the pipeline JSON structure.
from thinkstream.data.agent_protocol import format_memory_block as _shared_format_memory


def _format_memory_block(memory: Dict) -> str:
    """Format memory state as text. Delegates to shared agent_protocol."""
    return _shared_format_memory(memory)


def build_per_timestep_messages(sample: Dict, base_path: Path) -> List[Dict]:
    """Convert pipeline JSON sample to Qwen chat messages.

    Ordering (sft_engineering.md v3.0 §2.1, must not violate):
    <visual_window> + frames → <recalled_frames> + frames → <memory>
    → <recall_result> → <user_input>

    NOTE: This function handles pipeline-specific concerns (frame_paths vs
    video_path resolution, base_path joining) that the shared agent_protocol
    doesn't need to know about. The TEXT format (tags, JSON structure) is
    delegated to agent_protocol to guarantee train/inference identity.
    """
    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = 2.0  # AGENT_CHUNK_SEC

    # ── System prompt ──
    messages = [{"role": "system", "content": inp["system"]}]

    # ── User content (视频在前、文本在后，匹配 agent_protocol.build_user_content) ──
    user_content = []

    # ── Zone B: Visual window + video frames ──
    vw = inp["visual_window"]
    current_start = chunk_idx * chunk_sec
    current_end = current_start + chunk_sec
    vw_header = json.dumps({
        "start": vw["video_start"],
        "end": vw["video_end"],
        "frames": vw["frames"],
        "current_time": [current_start, current_end],
    })
    user_content.append({
        "type": "text",
        "text": f"<visual_window>{vw_header}</visual_window>",
    })

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)

    if "frame_paths" in vw:
        paths = [str(base_path / p) if not Path(p).is_absolute() else p
                 for p in vw["frame_paths"]]
        user_content.append({"type": "video", "video": paths})
    elif "frame_indices" in vw and video_path:
        logging.warning(
            f"Sample {sample.get('sample_id', '?')}: no frame_paths, "
            f"using video_start/end fallback"
        )
        user_content.append({
            "type": "video",
            "video": video_path,
            "video_start": vw["video_start"],
            "video_end": vw["video_end"],
        })
    else:
        raise ValueError(
            f"Sample {sample.get('sample_id', '?')}: visual_window has neither "
            f"frame_paths nor frame_indices. Cannot load video frames."
        )

    # ── Zone B continued: Recalled frames (recall_response only) ──
    if "recalled_frames" in inp:
        rf = inp["recalled_frames"]
        rf_header = json.dumps({
            "time_range": rf["time_range"],
            "source": rf.get("source", "historical_frames"),
            "n_frames": rf["n_frames"],
        })
        user_content.append({
            "type": "text",
            "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
        })
        if "frame_paths" in rf:
            paths = [str(base_path / p) if not Path(p).is_absolute() else p
                     for p in rf["frame_paths"]]
            user_content.append({"type": "video", "video": paths})
        elif video_path:
            logging.warning(
                f"Sample {sample.get('sample_id', '?')}: recalled_frames missing "
                f"frame_paths, using time range fallback"
            )
            user_content.append({
                "type": "video",
                "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })
        else:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: recalled_frames has "
                f"neither frame_paths nor video_path."
            )

    # ── Zone C: Memory block ──
    memory_text = _format_memory_block(inp["memory"])
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>",
    })

    # ── Zone C continued: Recall result (recall_response only) ──
    if inp.get("recall_result"):
        rr = inp["recall_result"]
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })

    # ── Zone D: User input ──
    if inp.get("user_input"):
        user_content.append({
            "type": "text",
            "text": f"\n<user_input>{inp['user_input']}</user_input>",
        })

    messages.append({"role": "user", "content": user_content})

    # ── Assistant output (training target) ──
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": sample["output"]}],
    })

    return messages


# ---------------------------------------------------------------------------
# Preprocessing: messages → model inputs with label masking
# ---------------------------------------------------------------------------

def preprocess_per_timestep(sample: Dict, processor) -> Dict:
    """Tokenize a per-timestep sample and mask labels.

    Only the assistant turn (output) contributes to loss.
    Uses processor.apply_chat_template for unified tokenization + vision.
    """
    base_path = Path(sample.get("data_path", "."))
    messages = build_per_timestep_messages(sample, base_path)

    # Tokenize with vision processing
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Label masking: IGNORE_INDEX everywhere, then unmask assistant span
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    # Find assistant span by token pattern.
    # These IDs are stable across Qwen2/2.5/3 tokenizer families
    # (base vocab tokens; adding special tokens appends, doesn't shift).
    ASSISTANT_TOKEN_ID = 77091   # "assistant" role token
    IM_END_TOKEN_ID = 151645     # <|im_end|>

    # Defensive check: verify token IDs match this tokenizer
    vocab = processor.tokenizer.get_vocab()
    assert vocab.get("assistant", -1) == ASSISTANT_TOKEN_ID or \
        ASSISTANT_TOKEN_ID in input_ids[0].tolist(), \
        f"Token ID mismatch: 'assistant' not at {ASSISTANT_TOKEN_ID} in this tokenizer"

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == ASSISTANT_TOKEN_ID:
            ans_start = pos + 2  # skip role token + newline
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != IM_END_TOKEN_ID:
                ans_end += 1
            if ans_end < L:
                # Unmask assistant tokens (include <|im_end|> + newline)
                labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # Per-sample loss weight
    sample_type = sample.get("sample_type", "silent")
    full_result["sample_weight"] = ACTION_WEIGHTS.get(sample_type, 1.0)

    return full_result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerTimestepDataset(Dataset):
    """Dataset for per-timestep agent SFT.

    Each sample is a single 2s chunk with memory state, visual window,
    optional recalled frames, and a single assistant output.
    """

    def __init__(self, processor, data_args):
        super().__init__()

        dataset_names = data_args.dataset_use.split(",")
        dataset_configs = data_list(dataset_names)
        rank0_print(f"Loading datasets: {dataset_configs}")

        # Select RoPE function by model type
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            raise ValueError(
                f"Unsupported model_type: {data_args.model_type}. "
                f"Supported: qwen2.5vl, qwen3vl"
            )

        # Load all samples
        all_samples = []
        for cfg in dataset_configs:
            path = cfg["annotation_path"]
            if path.endswith(".jsonl"):
                annotations = read_jsonl(path)
            else:
                with open(path) as f:
                    annotations = json.load(f)

            sampling_rate = cfg.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"  Sampled {len(annotations)} from {path}")

            for ann in annotations:
                ann["data_path"] = cfg["data_path"]
            all_samples.extend(annotations)

        # Filter overlong samples (P0-4: no silent truncation in collator)
        max_tokens = getattr(data_args, "max_sample_tokens", None)
        if max_tokens:
            before = len(all_samples)
            all_samples = [s for s in all_samples
                           if s.get("num_tokens", 0) < max_tokens]
            filtered = before - len(all_samples)
            if filtered > 0:
                rank0_print(f"  Filtered {filtered} samples > {max_tokens} tokens")

        rank0_print(f"Total training samples: {len(all_samples)}")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [s.get("num_tokens", 3500) for s in self.samples]

    @property
    def modality_lengths(self):
        return [s.get("num_tokens", 3500) for s in self.samples]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                return self._get_item(i)
            except Exception as e:
                logging.warning(f"[Attempt {attempt}] Failed sample {i}: {e}")
                time.sleep(0.5)
                if attempt == 2:
                    # Try next sample
                    i = min(i + 1, len(self.samples) - 1)

        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]

        # Tokenize + vision + label mask
        data_dict = preprocess_per_timestep(sample, self.processor)

        seq_len = data_dict["input_ids"][0].size(0)

        # Compute RoPE position IDs
        grid_thw = None
        if "image_grid_thw" in data_dict:
            g = data_dict["image_grid_thw"]
            grid_thw = [g] if not isinstance(g, list) else g

        video_grid_thw = None
        second_per_grid_ts = None
        if "video_grid_thw" in data_dict:
            vg = data_dict["video_grid_thw"]
            video_grid_thw = [vg] if not isinstance(vg, list) else vg

            # For Qwen2.5-VL: second_per_grid_ts controls temporal RoPE spacing.
            # Visual window and recalled frames should have different temporal
            # encodings (sft_engineering.md §3.3). For Qwen3-VL this is unused
            # (timestamps encode temporal info instead).
            default_spg = (
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            )
            n_video_entries = len(video_grid_thw)
            if n_video_entries == 2 and "recalled_frames" in sample.get("input", {}):
                # First entry = visual window, second = recalled frames
                rf = sample["input"]["recalled_frames"]
                rf_duration = rf["time_range"][1] - rf["time_range"][0]
                rf_n_frames = rf.get("n_frames", 4)
                rf_spg = rf_duration / max(rf_n_frames, 1)
                second_per_grid_ts = [default_spg, rf_spg]
            else:
                second_per_grid_ts = [default_spg] * n_video_entries

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        return data_dict


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

def pad_and_cat(tensor_list):
    max_length = max(t.shape[2] for t in tensor_list)
    padded = [
        torch.nn.functional.pad(t, (0, max_length - t.shape[2]), "constant", 1)
        for t in tensor_list
    ]
    return torch.cat(padded, dim=1)


@dataclass
class PerTimestepDataCollator:
    """Collate per-timestep samples into training batch.

    Adds per-sample loss weights (sft_engineering.md §5.2).
    Does NOT truncate — overlong samples filtered in Dataset init (P0-4).
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [inst[key] for inst in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX,
        )
        position_ids = pad_and_cat(position_ids)

        # P0-4: Do NOT truncate here. Overlong samples must be filtered in
        # Dataset init. Right-truncation would silently destroy output labels,
        # making the model train on input-only samples (all IGNORE_INDEX).
        max_len = self.tokenizer.model_max_length
        if input_ids.shape[1] > max_len:
            n_over = (input_ids.shape[1] > max_len).sum().item()
            logging.warning(
                f"PerTimestepDataCollator: {n_over} samples exceed max_length "
                f"{max_len}. These should have been filtered in Dataset init. "
                f"Check max_sample_tokens setting."
            )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
        }

        # Concatenate vision tensors
        videos = [inst["pixel_values_videos"] for inst in instances
                  if "pixel_values_videos" in inst]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [inst["video_grid_thw"] for inst in instances
                 if "video_grid_thw" in inst],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None

        images = [inst["pixel_values"] for inst in instances
                  if "pixel_values" in inst]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [inst["image_grid_thw"] for inst in instances
                 if "image_grid_thw" in inst],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        # Per-sample loss weights
        batch["sample_weights"] = torch.tensor(
            [inst.get("sample_weight", 1.0) for inst in instances],
            dtype=torch.float32,
        )

        return batch


# ---------------------------------------------------------------------------
# Module factory
# ---------------------------------------------------------------------------

def make_per_timestep_data_module(processor, data_args) -> Dict:
    """Create dataset + collator for per-timestep agent SFT."""
    train_dataset = PerTimestepDataset(processor, data_args)
    collator = PerTimestepDataCollator(processor.tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": collator,
    }
