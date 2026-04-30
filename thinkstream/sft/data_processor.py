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

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path: str) -> list:
    """Read .jsonl. v12.5 (2026-04-29): also reads .jsonl.gz transparently
    (gz form is committed-to-git for files >100MB GitHub limit, and pass4
    output for trajectory files is gzipped on the cluster path)."""
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _estimate_sample_tokens(sample: Dict) -> int:
    """Cheap text+vision token estimate (no tokenizer call).

    Used by Dataset.lengths for HF Trainer's group_by_length sampler and to
    drop overlong samples before they hit the GPU. Accuracy ±15% is fine —
    we only need correct ranking among samples.

    Handles BOTH schemas:
      (1) Messages format (post-pass5): sum text in content + count video frames
      (2) Flat format: parse input.{system,memory,queries,visual_window} fields

    Vision token cost per frame matches config.VISUAL_TOKENS_PER_CHUNK / 2
    = ~128 tokens at v12.5 resolution (min_pixels=100352, merge_size=2).
    The earlier 256 figure assumed a higher resolution (~150k pixels)
    and led to ~2× over-estimate, falsely flagging in-budget samples as
    overlong.
    """
    _VIS_TOK_PER_FRAME = 128  # matches config.VISUAL_TOKENS_PER_CHUNK / FRAMES_PER_CHUNK

    # ── Messages format ──
    if "messages" in sample:
        text_chars = 0
        n_frames = 0
        for msg in sample["messages"]:
            content = msg.get("content")
            if isinstance(content, str):
                text_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    t = item.get("type")
                    if t == "text":
                        text_chars += len(item.get("text", ""))
                    elif t == "video":
                        v = item.get("video")
                        if isinstance(v, list):
                            n_frames += len(v)
                        else:
                            # raw video w/ time range — estimate from interval
                            vs = item.get("video_start", 0)
                            ve = item.get("video_end", vs)
                            n_frames += max(1, int(ve - vs) * 2)  # FPS=2
        return text_chars // 3 + n_frames * _VIS_TOK_PER_FRAME

    # ── Flat format (legacy) ──
    inp = sample.get("input", {})
    out = sample.get("output", "")
    text_chars = (
        len(inp.get("system", ""))
        + len(out)
        + len(inp.get("user_input", "") or "")
    )
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
    n_frames = inp.get("visual_window", {}).get("frames", 12)
    rf = inp.get("recalled_frames")
    if rf:
        n_frames += rf.get("n_frames", 0)
    visual_tokens = n_frames * _VIS_TOK_PER_FRAME
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
        if hasattr(vp, "do_sample_frames"):
            vp.do_sample_frames = False

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


def build_per_timestep_messages_v12(sample: Dict, base_path: Path) -> List[Dict]:
    """v12.0: Build messages for the official Qwen tool-call protocol.

    Three sample shapes handled (controlled by pass3c-emitted fields):

    A. Single-turn (silent / response / lonely recall):
       sample["output"] = single assistant string. Messages = [system, user, assistant].

    B. Multi-turn recall (sample_type=='recall' with v12_assistant_turn_1/2):
       Two assistant turns sandwiching a tool turn. Messages =
         [system, user (chunk visual+memory+query),
          assistant (tool_call recall),
          tool (recall_result),
          assistant (final answer)]
       This implements the within-one-chunk agentic cycle (think→recall→
       result→think→answer) per docs/v12.0_protocol_migration_design.md §1.

    C. Inter-chunk compress (v12_inter_chunk=True):
       NO visual_window in user content (compression fires between visual
       timesteps). Messages = [system, user (memory + compress_trigger),
       assistant (tool_call compress)]. Reuses the same architecture but
       without the chunk's frames / recalled_frames sections.

    Differences from v11 (build_per_timestep_messages):
    - SYSTEM_PROMPT_V12 (concise; <tools> block rendered by chat_template
      via tools= parameter at apply time).
    - recall_result moved from user-inline text to a dedicated 'tool' role
      message in shape B (matches Qwen3-VL chat_template tool branch which
      nests <tool_response> inside the <|im_start|>user wrapper).
    """
    # v12.6: import canonical chunk_sec via agent_protocol (which already
    # falls back gracefully when scripts.agent_data_v5.config isn't on the
    # path — e.g. inference container). Earlier the direct
    # `from scripts.agent_data_v5.config import AGENT_CHUNK_SEC` would raise
    # ModuleNotFoundError when training was launched outside the project
    # root. Going through agent_protocol routes through the same fallback
    # chain SFT/eval/RL all use.
    from thinkstream.data.agent_protocol import (
        SYSTEM_PROMPT_V12,
        AGENT_CHUNK_SEC,
    )

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = float(AGENT_CHUNK_SEC)
    inter_chunk = bool(sample.get("v12_inter_chunk", False))
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_V12}]}
    ]

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)
    require_pre = bool(sample.get("_require_pre_extracted_frames", True))

    # ── User content ───────────────────────────────────────────────────
    user_content = []

    if not inter_chunk:
        # Visual window only present for visual timesteps (NOT inter-chunk
        # compress turns, where compression is a system event between two
        # visual chunks and consumes no new frames).
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

        # v12.5 fallback: pass4 flat files may omit frame_paths — infer from
        # video_id + frames count using the pre-extracted frame directory.
        if "frame_paths" not in vw and "frames" in vw:
            vid = sample.get("video_id", "")
            if vid:
                frame_dir = f"data/agent_v5/frames/{vid}"
                vw["frame_paths"] = [
                    f"{frame_dir}/frame_{i+1:06d}.jpg"
                    for i in range(vw["frames"])
                ]

        if "frame_paths" in vw:
            paths = [str(base_path / p) if not Path(p).is_absolute() else p
                     for p in vw["frame_paths"]]
            user_content.append({"type": "video", "video": paths})
        elif "frame_indices" in vw and video_path:
            if require_pre:
                raise ValueError(
                    f"Sample {sample.get('sample_id', '?')}: visual_window has no "
                    f"frame_paths. Pre-extract frames or set "
                    f"--require_pre_extracted_frames False."
                )
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": vw["video_start"], "video_end": vw["video_end"],
            })
        else:
            raise ValueError(
                f"Sample {sample.get('sample_id', '?')}: visual_window has neither "
                f"frame_paths nor frame_indices."
            )

    # Recalled frames stay in the FIRST user message ONLY for non-multi-turn
    # recall samples (legacy single-turn recall_response). For multi-turn
    # recall (shape B), recalled_frames are part of the tool turn payload
    # and rendered there, not in the prompt before the model emits anything.
    if "recalled_frames" in inp and not is_recall_multiturn and not inter_chunk:
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
        elif video_path and not require_pre:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })

    memory_text = _format_memory_block(inp["memory"])
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>" if not inter_chunk
        else f"<memory>\n{memory_text}\n</memory>",
    })

    queries = inp.get("queries", [])
    if queries and not inter_chunk:
        from thinkstream.data.agent_protocol import format_queries_block
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({"type": "text", "text": f"\n{queries_text}"})

    # Legacy (non-multi-turn) recall_result fallback. Multi-turn recall
    # samples render recall_result via the tool role below.
    if inp.get("recall_result") and not is_recall_multiturn and not inter_chunk:
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

    if inp.get("user_input"):
        # compress_trigger pre-injected by pass3c v12 (sample.input.user_input
        # contains "<compress_trigger range='a-b'/>" prefix).
        user_content.append({
            "type": "text",
            "text": (f"\n<user_input>{inp['user_input']}</user_input>"
                     if not inter_chunk else f"\n{inp['user_input']}"),
        })

    messages.append({"role": "user", "content": user_content})

    # ── Assistant turn(s) ──────────────────────────────────────────────
    if is_recall_multiturn:
        # Shape B: 2 assistant turns sandwiching a tool turn.
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })

        # Tool turn — recall_result wrapped in <tool_response> tags. The
        # Qwen3-VL chat_template renders this nested under <|im_start|>user
        # but loss-masked at training time (assistant span only contributes).
        rr = sample.get("recall_result") or {}
        rr_json = json.dumps({
            "source": rr.get("source", ""),
            "time": rr.get("time", ""),
            "text": rr.get("text_content", rr.get("text", "")),
        }, ensure_ascii=False)
        tool_payload = [{"type": "text", "text": rr_json}]

        # If the recall returned historical frames, attach them inside the
        # tool turn payload — model sees them as part of the tool response.
        rf = inp.get("recalled_frames")
        if rf:
            rf_header = json.dumps({
                "time_range": rf["time_range"],
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
            })
            tool_payload.append({
                "type": "text",
                "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
            })
            if "frame_paths" in rf:
                paths = [str(base_path / p) if not Path(p).is_absolute() else p
                         for p in rf["frame_paths"]]
                tool_payload.append({"type": "video", "video": paths})
            elif video_path and not require_pre:
                tool_payload.append({
                    "type": "video", "video": video_path,
                    "video_start": rf["time_range"][0],
                    "video_end": rf["time_range"][1],
                })
        messages.append({"role": "tool", "content": tool_payload})

        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_2"]}],
        })
    else:
        # Shape A or C: single assistant turn.
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
    """Tokenize a single SFT sample (messages format) and mask labels.

    Input contract (post-pass5): sample MUST contain a ``messages`` key
    (LLaMA-Factory ShareGPT format). Each message is {"role": str,
    "content": list-of-content-items}. Loss is computed only on the
    assistant turn(s) — exact same convention as VST / DeepEyesV2 /
    Qwen-VL official finetune (scan for <|im_start|>assistant ...
    <|im_end|>, that span gets loss).

    Always passes ``tools=TOOLS_SCHEMA`` to apply_chat_template so the
    system prompt auto-renders the ``<tools>`` block per Qwen3-VL spec.
    """
    base_path = Path(sample.get("data_path", "."))

    if "messages" not in sample:
        raise ValueError(
            f"Sample {sample.get('sample_id', '?')}: missing 'messages' key. "
            f"Run scripts/agent_data_v5/pass5_messages.py to convert "
            f"input/output samples to ShareGPT messages format."
        )
    messages = _resolve_video_paths(sample["messages"], base_path)

    from thinkstream.data.agent_protocol import TOOLS_SCHEMA
    template_kwargs = dict(
        tokenize=True, return_dict=True, return_tensors="pt", tools=TOOLS_SCHEMA,
    )

    full_result = processor.apply_chat_template(messages, **template_kwargs)

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

    # v12 allows 1 (silent/response/compress) or 2 (multi-turn recall)
    # assistant turns per sample. Every assistant turn contributes loss=1
    # (DeepEyesV2 multiturn_sft_dataset.py:170 pattern: gen-prompt prefix
    # masked, message body trained).
    if len(assistant_spans) not in {1, 2}:
        sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
        raise ValueError(
            f"Sample {sid}: expected 1 or 2 assistant turn(s), "
            f"found {len(assistant_spans)}. 1 turn for "
            f"silent/response/compress; 2 turns for recall multi-turn."
        )

    # Unmask all assistant turns (each can have its own [start, end] range).
    for ans_start, ans_end in assistant_spans:
        labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
    ans_start, ans_end = assistant_spans[0]

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # Vanilla CE per DeepEyesV2 multiturn_sft_dataset.py:170 — uniform
    # weight 1.0 across the assistant span.
    full_result["eval_meta"] = {
        "sample_type": sample.get("sample_type", "?"),
        "ans_start": ans_start,
        "ans_end": ans_end,
    }
    return full_result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerTimestepDataset(Dataset):
    """Dataset for per-timestep agent SFT.

    Each sample is a single 1s chunk (v12.5) with memory state, visual
    window (recent 16 chunks = 16s), optional recalled frames, and a
    single assistant output.
    """

    def __init__(self, processor, data_args, dataset_use_override: Optional[str] = None,
                 max_samples: Optional[int] = None):
        super().__init__()

        dataset_use = dataset_use_override if dataset_use_override is not None else data_args.dataset_use
        dataset_names = dataset_use.split(",")
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

        # v12.6: filter strictly to messages-format samples. preprocess_per_timestep
        # raises on missing 'messages' key (line ~448), so admitting flat-format
        # samples here would silently pass the dataset boundary then crash inside
        # training. Legacy flat datasets must be converted via pass5_messages.py
        # first; the original dual-schema filter caused confusing late-stage
        # crashes when a stale dataset path slipped through.
        def _is_valid_messages(s: Dict) -> bool:
            msgs = s.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return False
            return any(m.get("role") == "assistant" for m in msgs)

        before = len(all_samples)
        all_samples = [s for s in all_samples if _is_valid_messages(s)]
        empty_dropped = before - len(all_samples)
        if empty_dropped > 0:
            rank0_print(
                f"  Dropped {empty_dropped} samples — they had no 'messages' "
                f"key or no assistant turn. If this is a flat-format dataset, "
                f"convert via:  python -m scripts.agent_data_v5.pass5_messages"
            )

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

        # v11.3: per-sample memory uniqueness (used by class-balanced sampler
        # when --unique_think_weight is enabled). Down-weights samples whose
        # memory snapshot has many duplicate thinks — typical for static-scene
        # videos where the teacher correctly reports "scene unchanged" but
        # those repeated entries don't add training value.
        for s in all_samples:
            mem = (s.get("input") or {}).get("memory") or {}
            thinks = mem.get("recent_thinks") or []
            texts = []
            for t in thinks:
                if isinstance(t, dict):
                    texts.append(t.get("text", ""))
                elif isinstance(t, str):
                    texts.append(t)
            if not texts:
                # No thinks yet (early chunks): treat as fully-unique so
                # warmup samples don't get accidentally down-weighted.
                s["_unique_rate"] = 1.0
            else:
                s["_unique_rate"] = len(set(texts)) / len(texts)

        # Optional eval-side cap: keep in-loop eval fast on large val pools.
        # Deterministic subsample (seeded RNG) so train logs stay comparable
        # across runs.
        if max_samples is not None and max_samples > 0 and len(all_samples) > max_samples:
            rng = random.Random(0)
            all_samples = rng.sample(all_samples, max_samples)
            rank0_print(f"  Subsampled eval set to {max_samples}")

        rank0_print(f"Total samples: {len(all_samples)}")

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

        # Tokenize + vision + label mask. apply_chat_template always uses
        # tools=TOOLS_SCHEMA (the v12 protocol).
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

        # Pad per-token loss weights to same shape as labels.
        # Pad value = 0.0: padding tokens are masked by valid_mask anyway,
        # but explicit 0 avoids accidentally weighting padding if mask logic
        # ever changes.
        token_loss_weights = [
            inst["token_loss_weight"].squeeze(0) for inst in instances
            if "token_loss_weight" in inst
        ]
        if len(token_loss_weights) == len(instances):
            token_loss_weight = torch.nn.utils.rnn.pad_sequence(
                token_loss_weights, batch_first=True, padding_value=0.0,
            )
        else:
            token_loss_weight = None

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
        if token_loss_weight is not None:
            batch["token_loss_weight"] = token_loss_weight

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

        # Per-sample metadata (audit logging only; popped before model.forward)
        batch["sample_meta"] = [inst.get("sample_meta", {}) for inst in instances]

        # Eval-time accuracy probes (popped before model.forward in trainer)
        batch["eval_meta"] = [inst.get("eval_meta", {}) for inst in instances]

        return batch


# ---------------------------------------------------------------------------

def make_per_timestep_data_module(processor, data_args) -> Dict:
    """Create dataset + collator for per-timestep agent SFT.

    Builds an eval_dataset when DataArguments.eval_dataset_use is set —
    typically `stream_agent_val` (held-out video-disjoint pool). The HF
    Trainer then runs eval on this every --eval_steps to surface
    overfitting in real time.
    """
    train_dataset = PerTimestepDataset(processor, data_args)

    eval_dataset = None
    eval_use = getattr(data_args, "eval_dataset_use", None)
    if eval_use:
        rank0_print(f"Building eval_dataset from: {eval_use}")
        eval_dataset = PerTimestepDataset(
            processor,
            data_args,
            dataset_use_override=eval_use,
            max_samples=getattr(data_args, "eval_max_samples", None),
        )

    collator = PerTimestepDataCollator(processor.tokenizer)

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
