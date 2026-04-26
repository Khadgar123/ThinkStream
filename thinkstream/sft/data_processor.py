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

# Per-sample loss weights by action type.
#
# Design principle (see data_batch1_plan.md §5.3):
# - silent/response: timing is DETERMINISTIC (answer visible or not).
#   SFT must strictly learn correct timing → HIGH weight.
# - recall/compress: timing is NON-DETERMINISTIC (no single correct moment).
#   SFT teaches mechanism (format, query/summary quality), NOT timing.
#   Timing optimization is left to RL/GRPO → LOW weight.
ACTION_WEIGHTS = {
    # ── Core behaviors (SFT teaches mechanism + timing) ──
    "response": 1.5,            # Answering visible questions — highest value
    "silent": 1.0,              # Default silent (with active queries — teaches restraint)

    # ── Recall mechanism (SFT teaches format, RL optimizes timing) ──
    "recall_query": 0.8,        # Recall query format learning
    "recall_response": 1.0,     # Post-recall response (deterministic: evidence arrived)
    "recall_silent": 0.8,       # Post-recall silent (recall failed)
    "proactive_recall_query": 0.8,
    "proactive_recall_silent": 0.8,

    # ── Compression mechanism (SFT teaches summary quality) ──
    "compress": 0.8,
    "merge_compress": 0.8,
}


def _get_sample_weight(sample: Dict) -> float:
    """Compute loss weight based on sample_type + context + trajectory role.

    Goes beyond flat ACTION_WEIGHTS: distinguishes silent subtypes AND
    base sample roles to ensure critical training signals are not buried.

    Key insight: base samples at evidence_anchor chunks (support_chunks ± 2)
    teach the model to OBSERVE and RETAIN facts needed for future recall.
    These must have high weight — otherwise the model never learns good
    memory formation, making recall decisions unreliable downstream.
    """
    sample_type = sample.get("sample_type", "silent")
    sequence_type = sample.get("sequence_type", "")

    # Use ACTION_WEIGHTS for non-silent types
    if sample_type != "silent":
        return ACTION_WEIGHTS.get(sample_type, 1.0)

    # Silent subtypes — different training value:
    queries = sample.get("queries") or sample.get("input", {}).get("queries", [])

    if sequence_type == "base":
        # Base samples carry a base_role from pass3c that indicates
        # WHY this chunk was selected for training
        base_role = sample.get("base_role", "")

        if base_role == "evidence_anchor":
            # support_chunks ± 2: model must learn to observe and retain
            # facts that will be needed for future recall questions.
            # High weight regardless of queries — memory formation is
            # critical even before any question arrives.
            return 1.2
        elif base_role == "compress_boundary":
            # Chunks around compression events: critical for learning
            # what to preserve and what to discard during compression.
            return 1.0
        elif base_role == "question_window":
            # Chunks around Q&A events: context for decision boundaries.
            return 0.8 if queries else 0.5
        elif base_role == "warmup":
            # Cold-start chunks: empty memory, minimal signal.
            return 0.3
        elif base_role == "patrol":
            # Long-silent stretches: teaches sustained silence.
            return 0.8 if queries else 0.3
        else:
            # Legacy samples without base_role (backward compat)
            return 0.8 if queries else 0.3

    elif sequence_type == "event_watch":
        return 1.0        # "Event hasn't happened, keep watching" — teaches patience
    elif sequence_type in ("immediate_response", "recall_success", "recall_fail_then_found"):
        return 0.5        # Post-action recovery — transitional
    elif sequence_type == "multi_response":
        return 0.8        # "No new change, stay silent" — teaches selective response

    return 0.5  # Default silent

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
    # Queries zone (past Q&A tracking)
    "<queries>", "</queries>",
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


def _estimate_sample_tokens(sample: Dict) -> int:
    """Cheap text+vision token estimate (no tokenizer call).

    Used by Dataset.lengths for HF Trainer's group_by_length sampler and to
    drop overlong samples before they hit the GPU. Accuracy ±15% is fine —
    we only need correct ranking among samples.

    Vision token cost per frame at the configured resolution is ~256 tokens
    (Qwen2.5-VL/Qwen3-VL with merge_size=2 at ~150k pixels).
    """
    inp = sample.get("input", {})
    out = sample.get("output", "")

    # ── Text characters → ~3 chars/token for English+JSON (Qwen tokenizer) ──
    text_chars = (
        len(inp.get("system", ""))
        + len(out)
        + len(inp.get("user_input", "") or "")
    )
    # memory is a dict; serialize to estimate
    mem = inp.get("memory", {}) or {}
    for seg in mem.get("compressed_segments", []):
        text_chars += len(json.dumps(seg, ensure_ascii=False))
    for t in mem.get("recent_thinks", []):
        text_chars += len(t) if isinstance(t, str) else len(json.dumps(t, ensure_ascii=False))
    for q in inp.get("queries", []) or []:
        text_chars += len(json.dumps(q, ensure_ascii=False))
    rr = inp.get("recall_result")
    if rr:
        text_chars += len(json.dumps(rr, ensure_ascii=False))

    text_tokens = text_chars // 3

    # ── Vision tokens ──
    n_frames = inp.get("visual_window", {}).get("frames", 12)
    rf = inp.get("recalled_frames")
    if rf:
        n_frames += rf.get("n_frames", 0)
    visual_tokens = n_frames * 256

    return text_tokens + visual_tokens


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

    require_pre = bool(sample.get("_require_pre_extracted_frames", True))
    if "frame_paths" in vw:
        paths = [str(base_path / p) if not Path(p).is_absolute() else p
                 for p in vw["frame_paths"]]
        user_content.append({"type": "video", "video": paths})
    elif "frame_indices" in vw and video_path:
        if require_pre:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: visual_window has no "
                f"frame_paths (only frame_indices). Pre-extract frames or set "
                f"--require_pre_extracted_frames False (NOT recommended for "
                f"real training: online decoding is ~50× slower)."
            )
        logging.warning(
            f"Sample {sample.get('sample_id', '?')}: no frame_paths, "
            f"using video_start/end fallback (slow online decode)"
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
            if require_pre:
                raise ValueError(
                    f"Sample {sample.get('sample_id', '?')}: recalled_frames "
                    f"has no frame_paths. Pre-extract recall frames or disable "
                    f"--require_pre_extracted_frames."
                )
            logging.warning(
                f"Sample {sample.get('sample_id', '?')}: recalled_frames missing "
                f"frame_paths, using time range fallback (slow online decode)"
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

    # ── Zone Q: Queries (past Q&A context, independent of memory) ──
    queries = inp.get("queries", [])
    if queries:
        from thinkstream.data.agent_protocol import format_queries_block
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({
                "type": "text",
                "text": f"\n{queries_text}",
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

def _resolve_video_paths(messages: List[Dict], base_path: Path) -> List[Dict]:
    """Resolve relative video paths in messages to absolute paths."""
    resolved = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    item = dict(item)
                    vp = item.get("video", "")
                    if isinstance(vp, str) and vp and not Path(vp).is_absolute():
                        item["video"] = str(base_path / vp)
                new_content.append(item)
            msg = {**msg, "content": new_content}
        resolved.append(msg)
    return resolved


# Cache so we resolve once per tokenizer (rather than per sample).
_CHAT_TEMPLATE_ID_CACHE: Dict[int, tuple] = {}


def _resolve_chat_template_ids(tokenizer) -> tuple:
    """Resolve (assistant_role_token_id, im_end_token_id) from the tokenizer.

    Fails loudly with a precise diagnosis if the chat template doesn't
    contain the expected tokens, so a tokenizer drift surfaces immediately
    instead of silently producing wrong loss masks.
    """
    cache_key = id(tokenizer)
    cached = _CHAT_TEMPLATE_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vocab = tokenizer.get_vocab()
    im_end_id = vocab.get("<|im_end|>")
    if im_end_id is None:
        raise RuntimeError(
            "Tokenizer drift: '<|im_end|>' not in vocab. "
            "SFT loss masking depends on Qwen chat-template tokens."
        )

    # The Qwen chat template emits "<|im_start|>assistant\n" — the role token
    # immediately follows <|im_start|>. Probe by encoding the template and
    # require exactly 2 tokens — anything else means "assistant" got split
    # into sub-tokens and our mask logic would point at the wrong span.
    probe_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    im_start_id = vocab.get("<|im_start|>")
    if im_start_id is None or im_start_id not in probe_ids:
        raise RuntimeError(
            f"Tokenizer drift: probe ids {probe_ids!r} do not contain "
            f"<|im_start|> ({im_start_id}). Cannot locate assistant role token."
        )
    if len(probe_ids) != 2:
        raise RuntimeError(
            f"Tokenizer drift: '<|im_start|>assistant' tokenized to "
            f"{len(probe_ids)} tokens ({probe_ids!r}); expected exactly 2 "
            f"([<|im_start|>, assistant]). The 'assistant' role token may not "
            f"be registered as a single chat-template token in this tokenizer."
        )
    idx = probe_ids.index(im_start_id)
    assistant_id = probe_ids[idx + 1]

    _CHAT_TEMPLATE_ID_CACHE[cache_key] = (assistant_id, im_end_id)
    return assistant_id, im_end_id


def preprocess_per_timestep(sample: Dict, processor) -> Dict:
    """Tokenize a per-timestep sample and mask labels.

    Only the assistant turn (output) contributes to loss.
    Uses processor.apply_chat_template for unified tokenization + vision.

    Accepts two formats:
    - Messages format (pipeline v5): sample["messages"] used directly
    - Legacy format: sample["input"]/["output"] built via build_per_timestep_messages()
    """
    base_path = Path(sample.get("data_path", "."))

    if "messages" in sample:
        # Pipeline v5: messages already constructed, resolve paths and use directly
        messages = _resolve_video_paths(sample["messages"], base_path)
    else:
        # Legacy: build messages from input/output structure
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
    # These IDs are stable across Qwen2/2.5/3 tokenizer families, but we
    # resolve them dynamically from the actual tokenizer to avoid silent
    # mask drift if the upstream vocab ever shifts.
    ASSISTANT_TOKEN_ID, IM_END_TOKEN_ID = _resolve_chat_template_ids(
        processor.tokenizer
    )

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    assistant_spans: List[tuple] = []
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == ASSISTANT_TOKEN_ID:
            ans_start = pos + 2  # skip role token + newline
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != IM_END_TOKEN_ID:
                ans_end += 1
            if ans_end < L:
                assistant_spans.append((ans_start, ans_end))
                pos = ans_end
        pos += 1

    # Per-timestep design: exactly one assistant turn per sample.
    # Multiple turns (or zero) means data corruption — fail loudly so we
    # don't silently train on the wrong span.
    if len(assistant_spans) != 1:
        sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
        raise ValueError(
            f"Sample {sid}: expected exactly 1 assistant turn, "
            f"found {len(assistant_spans)}. Per-timestep samples must have "
            f"a single assistant output."
        )
    ans_start, ans_end = assistant_spans[0]
    # Unmask assistant tokens (include <|im_end|> + newline)
    labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # Per-sample loss weight (context-aware, not just sample_type)
    full_result["sample_weight"] = _get_sample_weight(sample)

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
                ann["_require_pre_extracted_frames"] = bool(
                    getattr(data_args, "require_pre_extracted_frames", True)
                )
            all_samples.extend(annotations)

        # Estimate num_tokens for every sample (used for length-based filtering
        # AND HF Trainer's group_by_length sampler). Skipping this leaves every
        # sample with default 3500 → batches are wildly heterogeneous → padding
        # waste + silent overflow.
        for s in all_samples:
            if "num_tokens" not in s:
                s["num_tokens"] = _estimate_sample_tokens(s)

        # Filter overlong samples (P0-4: no silent truncation in collator)
        max_tokens = getattr(data_args, "max_sample_tokens", None)
        if max_tokens:
            before = len(all_samples)
            all_samples = [
                s for s in all_samples if s.get("num_tokens", 0) < max_tokens
            ]
            filtered = before - len(all_samples)
            if filtered > 0:
                rank0_print(f"  Filtered {filtered} overlong (>{max_tokens} tok)")

        rank0_print(f"Total training samples: {len(all_samples)}")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.samples = all_samples

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        # num_tokens is now populated in __init__ for every sample.
        return [s["num_tokens"] for s in self.samples]

    @property
    def modality_lengths(self):
        return [s["num_tokens"] for s in self.samples]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Try sample i, then walk forward up to MAX_LOOKAHEAD if it keeps
        # failing. Avoids unbounded recursion on systemic data corruption.
        MAX_LOOKAHEAD = 16
        last_err = None
        for offset in range(MAX_LOOKAHEAD):
            j = (i + offset) % len(self.samples)
            try:
                return self._get_item(j)
            except Exception as e:
                last_err = e
                if offset == 0:
                    logging.warning(f"[sample {j}] failed: {e}")
                continue
        raise RuntimeError(
            f"All {MAX_LOOKAHEAD} samples after idx {i} failed to load. "
            f"Last error: {last_err}"
        )

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
            # Check recalled_frames in both messages format and legacy format
            rf_meta = sample.get("recalled_frames_meta") or \
                (sample.get("input", {}).get("recalled_frames"))
            if n_video_entries == 2 and rf_meta:
                # First entry = visual window, second = recalled frames
                rf_duration = rf_meta["time_range"][1] - rf_meta["time_range"][0]
                rf_n_frames = rf_meta.get("n_frames", 4)
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

        # Audit metadata — passed through collator to trainer for per-sample logs.
        data_dict["sample_meta"] = {
            "sample_id": sample.get("sample_id") or sample.get("trajectory_id"),
            "video_id": sample.get("video_id"),
            "chunk_idx": sample.get("chunk_idx"),
            "sample_type": sample.get("sample_type"),
            "action": sample.get("action"),
            "sequence_type": sample.get("sequence_type"),
            "base_role": sample.get("base_role"),
        }

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

        # Per-sample metadata (audit logging only; popped before model.forward)
        batch["sample_meta"] = [inst.get("sample_meta", {}) for inst in instances]

        return batch


# ---------------------------------------------------------------------------
# Class-balanced distributed sampler
# ---------------------------------------------------------------------------

class ClassBalancedDistributedSampler(torch.utils.data.Sampler):
    """Sample indices with weights inversely proportional to sample_type freq.

    Why we need this:
      Empirically samples are silent 70% / compress 25% / recall 4% / response
      ~3-10% (after pass3c fix). A uniform sampler trains the model overwhelmingly
      on `silent`, leaving recall/response under-fit. WeightedRandomSampler with
      inv-freq weights gives every action class a comparable per-batch share.

    Distributed-aware:
      Each rank draws (num_samples_per_epoch / world_size) indices independently
      using its own seed-offset RNG. Indices may overlap across ranks — this is
      fine because `WeightedRandomSampler` is stochastic and the gradient mean
      across ranks is unbiased w.r.t. the sampling distribution. (Strict
      partitioning gives tighter epochs but loses the inv-freq benefit on the
      tail; the looser variant is what BalancedDistributedSampler does in
      Hugging Face's own RLHF stack.)

    Args:
        sample_types: list[str] — per-sample class label
        num_samples: int — total draws per epoch (across all ranks)
        rank, world_size: standard distributed args (auto-detected if None)
        seed: base RNG seed; each rank adds its rank to derive its own
        replacement: True (default) — match WeightedRandomSampler semantics
        smoothing: optional damping on inv-freq weights to avoid over-emphasizing
            ultra-rare classes (e.g., one freak class). 1.0 = pure inv-freq,
            0.5 = √(inv-freq), 0.0 = uniform. Default 0.7.
    """

    def __init__(
        self,
        sample_types: List[str],
        num_samples: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
        smoothing: float = 0.7,
    ):
        from collections import Counter
        if rank is None or world_size is None:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank() if rank is None else rank
                    world_size = dist.get_world_size() if world_size is None else world_size
            except Exception:
                pass
        self.rank = rank if rank is not None else 0
        self.world_size = world_size if world_size is not None else 1

        self.n_total = len(sample_types)
        self.num_samples = num_samples or self.n_total
        # Per-rank sample count
        self.per_rank = self.num_samples // self.world_size
        self.seed = seed
        self.replacement = replacement
        self.epoch = 0

        # Inverse-frequency weights with smoothing
        cls_counts = Counter(sample_types)
        n_classes = len(cls_counts)
        cls_weights = {}
        for cls, cnt in cls_counts.items():
            inv = self.n_total / (n_classes * cnt)  # uniform-targeting weight
            cls_weights[cls] = inv ** smoothing
        # Normalize so mean weight = 1.0 (preserves overall scale)
        mean_w = sum(cls_weights.values()) / len(cls_weights)
        cls_weights = {k: v / mean_w for k, v in cls_weights.items()}
        self._cls_weights_summary = cls_weights  # for logging

        self.weights = torch.tensor(
            [cls_weights[t] for t in sample_types], dtype=torch.double
        )

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000 + self.rank)
        idxs = torch.multinomial(
            self.weights, self.per_rank, replacement=self.replacement, generator=g
        ).tolist()
        return iter(idxs)

    def __len__(self) -> int:
        return self.per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


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
