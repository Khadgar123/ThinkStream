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

# Token-span loss weights (v11.2).
#
# Streaming-video literature consensus (8/8 papers from 2024-2026 surveyed:
# VideoLLM-online, VideoLLM-MoD, MMDuet, ViSpeak, Dispider, StreamMind,
# Eyes Wide Open, State-Token Unified):
#   "Decision tokens are vastly outnumbered by description tokens in
#   typical streaming-agent training data; uniform per-token CE
#   under-trains the action behavior."
#
# Two main mitigation patterns: (A) token-level weighted CE on the action
# span [VideoLLM-online's indicator mask, StreamMind's W_s = 10·P,
# Eyes Wide Open's ω·L_LM], or (B) auxiliary BCE classification head
# [MMDuet, ViSpeak, Dispider]. We use (A) because it integrates with the
# existing WeightedSFTTrainer.compute_loss path (token_loss_weight
# infrastructure already in place at trainer.py).
#
# Weights below give the action keyword span ~17% of the effective
# gradient (vs ~2% under uniform CE) on a typical response sample,
# while keeping <think> at non-zero weight so the model still learns
# observation grounding (don't mask think entirely — see VideoLLM-MoD).
SPAN_WEIGHTS = {
    # v11.3 (post-eval-debug): think bumped 0.3 → 0.6 because eval ckpt
    # showed think collapsing to 280-300 token repetitive boilerplate at
    # inference time — span weight 0.3 was too low to teach brevity / pin
    # the length distribution to the SFT-baked 130-token cap.
    "think": 0.6,
    "action": 8.0,       # Core decision (silent/response/recall/compress)
    "response": 2.0,     # Answer content — main behavioral output
    "query": 2.0,        # Recall query content
    "summary": 1.0,      # Summary content (RL/GDPO optimizes further)
    "default": 1.0,      # Tokens outside any tracked span (e.g., between </think><action>)
}

# Per-sample loss weights by action type.
#
# v11.3 (post-eval-debug rebalance):
# Original design left compress/recall at 0.8 with the rationale "timing
# is non-deterministic, leave to RL." But the v11.2 SFT ckpt collapses to
# silent at compress_trigger time — RL can't improve format/timing if SFT
# never produces compress action at all. Bumped compress + recall_query
# to compete with silent's 70% data baseline so the model learns the
# format reliably. RL still refines exact timing; SFT now teaches the
# trigger→action binding hard.
#
# Original (deprecated): compress=0.8, recall_query=0.8, recall_silent=0.8
ACTION_WEIGHTS = {
    # ── Core behaviors (SFT teaches mechanism + timing) ──
    "response": 1.5,            # Answering visible questions — highest value
    # v11.4 (Path B from v11.3 postmortem): silent 1.0 → 1.2. The first
    # v11.3 SFT run showed silent_acc 99% → 86% — silent was crowded out
    # by compress (compress×7.2 vs silent gradient share). Bumping silent
    # from 1.0 to 1.2 + reducing compress 2.5→1.8 brings the ratio back
    # to ~3.6×, still favoring compress (still need it strong) but no
    # longer crushing silent.
    "silent": 1.2,

    # ── Recall mechanism (SFT teaches format, RL optimizes timing) ──
    "recall_query": 1.5,
    "recall_response": 1.5,
    "recall_silent": 1.0,
    "proactive_recall_query": 1.5,
    "proactive_recall_silent": 1.0,

    # ── Compression mechanism (SFT teaches summary quality) ──
    "compress": 1.8,            # v11.4 (Path B): 2.5 → 1.8 — see silent comment
    "merge_compress": 1.8,      # mirror compress
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
            # v11.3: no_query branch 0.5 → 1.0. Original under-weighted
            # "no question pending → still silent" examples; eval ckpt
            # learned to leak <action>response</action> at silent chunks.
            return 0.8 if queries else 1.0
        elif base_role == "warmup":
            # Cold-start chunks: empty memory, minimal signal.
            return 0.3
        elif base_role == "patrol":
            # Long-silent stretches: teaches sustained silence.
            # v11.3: no_query branch 0.3 → 0.8 (sustained silence is core).
            return 0.8
        else:
            # Legacy samples without base_role (backward compat).
            # v11.3: no_query branch 0.3 → 0.8 (same rationale as patrol).
            return 0.8

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
        if hasattr(vp, "do_sample_frames"):
            vp.do_sample_frames = False

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


def smart_init_special_token_embeddings(model, processor, special_tokens: list):
    """v11.4 fix: copy natural-word embeddings into new special-token slots.

    Background — the v11.3 SFT run produced eval/action_acc=0.98
    teacher-forced but free-generation outputs `</think>responseThe...`
    (skipping <action>/</action>/<response> tags entirely). Root cause:
    HF's default resize_token_embeddings initializes new rows as the
    mean of existing embeddings, producing very small magnitude. Even
    with SPAN_WEIGHTS["action"]=8.0 the gradient can't move the new
    embeddings far enough in 1 epoch — at sampling time, dot products
    against the well-trained natural-word "response" beat the small-
    magnitude `<action>` token, so the model emits "response" as plain
    text right after `</think>` instead of the structural token.

    Fix: for each new tag like `<action>`, find the underlying natural
    word ("action") in the existing tokenizer and copy that word's
    embedding into the new slot. Magnitude is now in the normal range
    from step 0; the structural tokens compete fairly with natural
    English right out of the gate.

    Tags handled:
      `<X>`   ←  embedding of word X
      `</X>`  ←  same as `<X>` (no separate "/X" word in vocab)
      multi-word tags (e.g. `<recall_result>`) split on `_` and average

    Call AFTER `model.resize_token_embeddings(...)` and BEFORE training.
    Side effect: also writes the same vector into the LM-head row for
    each new token (necessary when tie_word_embeddings=False; harmless
    when tied because the LM head IS the input embedding).
    """
    import re as _re
    import torch as _torch

    tokenizer = processor.tokenizer
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()  # may be None or tied

    initialized = []
    skipped = []
    for tag in special_tokens:
        tag_id = tokenizer.convert_tokens_to_ids(tag)
        if tag_id is None or tag_id == tokenizer.unk_token_id:
            skipped.append((tag, "tag_not_in_vocab"))
            continue
        # Strip < > / and split on _ to get inner word(s).
        inner = _re.sub(r"[<>/]", "", tag).strip()
        if not inner:
            skipped.append((tag, "empty_inner"))
            continue
        # Tokenize the inner word and average if multi-token.
        word_ids = tokenizer.encode(inner, add_special_tokens=False)
        if not word_ids:
            skipped.append((tag, f"no_tokens_for_{inner!r}"))
            continue
        # Underscore-separated names (e.g. "recall_result") get word-by-
        # word averaging so the seed embedding represents the concept.
        if "_" in inner:
            parts = inner.split("_")
            part_vecs = []
            for p in parts:
                pids = tokenizer.encode(p, add_special_tokens=False)
                if pids:
                    part_vecs.append(embed.weight.data[pids].mean(dim=0))
            if not part_vecs:
                skipped.append((tag, "no_part_tokens"))
                continue
            seed = _torch.stack(part_vecs).mean(dim=0)
        else:
            seed = embed.weight.data[word_ids].mean(dim=0)

        with _torch.no_grad():
            embed.weight.data[tag_id].copy_(seed)
            if lm_head is not None and lm_head is not embed and \
               lm_head.weight.shape[0] >= tag_id + 1:
                lm_head.weight.data[tag_id].copy_(seed)
        initialized.append((tag, inner, tag_id))

    rank0_print(
        f"[smart_init] initialized {len(initialized)} special-token embeddings "
        f"from natural words; skipped {len(skipped)}"
    )
    for tag, src, tid in initialized[:8]:
        rank0_print(f"  {tag} (id={tid}) ← embed_of({src!r})")
    if skipped:
        for tag, reason in skipped[:4]:
            rank0_print(f"  ⚠ skipped {tag}: {reason}")
    return {"initialized": initialized, "skipped": skipped}


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
    # Qwen3VL processor requires all message content to be list-of-dicts format
    messages = [{"role": "system", "content": [{"type": "text", "text": inp["system"]}]}]

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
    from thinkstream.data.agent_protocol import SYSTEM_PROMPT_V12

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = 2.0  # AGENT_CHUNK_SEC
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


# Cache so we resolve once per tokenizer (rather than per sample).
_SPAN_TOKEN_CACHE: Dict[int, Dict[str, Dict]] = {}


def _resolve_span_token_ids(tokenizer) -> Dict[str, Dict]:
    """Resolve open/close token SEQUENCES for each loss-weighted span.

    v11.4 industry mode: agent tags are NOT registered as single-token
    vocab entries by default. The base BPE tokenizes `<action>` as a
    multi-token sequence (typically `<` + `action` + `>` for Qwen, but
    could be other splits for different tokenizers). We resolve each
    span's open/close as the SEQUENCE the tokenizer produces — this
    works uniformly whether the tag is single-token (legacy mode after
    register_special_tokens) or multi-token (industry mode).

    Returns: {"think": {"open_seq": List[int], "close_seq": List[int],
                        "weight": float, "open": int (=open_seq[0]),
                        "close": int (=close_seq[-1])}, ...}

    Both `open_seq` and the legacy single-id `open` keys are present
    so existing call sites that read `info["open"]` still work
    (they'll see the FIRST token of the open sequence, which is the
    natural anchor — e.g., `<` for `<action>`).

    Skips spans where the tokenizer can't encode the tag string (rare;
    indicates a fundamental tokenizer issue).
    """
    cache_key = id(tokenizer)
    cached = _SPAN_TOKEN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spans_def = [
        ("think",    "<think>",    "</think>"),
        ("action",   "<action>",   "</action>"),
        ("response", "<response>", "</response>"),
        ("query",    "<query>",    "</query>"),
        ("summary",  "<summary>",  "</summary>"),
    ]
    result: Dict[str, Dict] = {}
    failed: List[str] = []
    for name, open_str, close_str in spans_def:
        open_seq = tokenizer.encode(open_str, add_special_tokens=False)
        close_seq = tokenizer.encode(close_str, add_special_tokens=False)
        if not open_seq or not close_seq:
            failed.append(name)
            continue
        result[name] = {
            "open_seq": open_seq,
            "close_seq": close_seq,
            "open": open_seq[0],          # legacy compat: first token of seq
            "close": close_seq[-1],       # legacy compat: last token of seq
            "weight": SPAN_WEIGHTS[name],
        }

    if failed:
        rank0_print(
            f"[span-weight] tokenizer can't encode spans {failed} — "
            f"those spans fall back to default weight 1.0."
        )
    multi_token = [name for name, info in result.items()
                   if len(info["open_seq"]) > 1]
    summary = ", ".join(
        f"{k}(w={v['weight']},open_len={len(v['open_seq'])})"
        for k, v in result.items()
    )
    rank0_print(f"[span-weight] active spans: {summary}")
    if multi_token:
        rank0_print(
            f"[span-weight] multi-token spans (industry mode): {multi_token} — "
            f"sequence matching enabled in _build_token_loss_weight."
        )

    _SPAN_TOKEN_CACHE[cache_key] = result
    return result


def _seq_match_at(haystack: List[int], pos: int, needle: List[int]) -> bool:
    """Check if needle starts at pos in haystack."""
    if pos + len(needle) > len(haystack):
        return False
    return haystack[pos: pos + len(needle)] == needle


def _build_token_loss_weight(
    input_ids_flat: List[int],
    ans_start: int,
    ans_end: int,
    span_ids: Dict[str, Dict],
    seq_len: int,
) -> torch.Tensor:
    """Walk assistant span and assign per-token weight by enclosing tag.

    v11.4 industry mode: span open/close are token SEQUENCES (multi-token
    when tags aren't registered as single tokens, length-1 when they are).
    Sequence-matching at each position so the same code handles both
    industry mode (`<action>` = `<` + `action` + `>` = 3 tokens) and
    legacy mode (`<action>` = single token id).

    Tokens outside [ans_start, ans_end+1] get default weight (zeroed by
    valid_mask in compute_loss anyway, but kept consistent for safety).

    Open-tag sub-tokens are all assigned the NEW span's weight; close-tag
    sub-tokens are assigned the CURRENT span's weight (so the closing
    tag itself counts as part of its own span). This matches VideoLLM-
    online's indicator-on-decision-tokens convention, generalized to
    multi-token tags.
    """
    weight = torch.full(
        (1, seq_len), SPAN_WEIGHTS["default"], dtype=torch.float
    )
    if not span_ids:
        return weight

    # Pre-extract sequences for fast checks. Sort opens by length DESC so
    # if any one tag's open seq is a prefix of another's (rare but
    # possible), the longer match wins.
    open_specs = sorted(
        [(name, info["open_seq"]) for name, info in span_ids.items()],
        key=lambda x: -len(x[1]),
    )

    current_span: Optional[str] = None
    current_close_seq: Optional[List[int]] = None
    upper = min(ans_end + 2, seq_len)
    i = ans_start
    while i < upper:
        # 1. If currently in a span, check for its close sequence first.
        if current_span is not None and current_close_seq is not None:
            if _seq_match_at(input_ids_flat, i, current_close_seq):
                w = span_ids[current_span]["weight"]
                end = min(i + len(current_close_seq), upper)
                for j in range(i, end):
                    weight[0, j] = w
                i = end
                current_span = None
                current_close_seq = None
                continue

        # 2. Try to match an open sequence (only if not already in a span).
        matched = None
        if current_span is None:
            for name, seq in open_specs:
                if _seq_match_at(input_ids_flat, i, seq):
                    matched = (name, seq)
                    break

        if matched is not None:
            name, seq = matched
            w = span_ids[name]["weight"]
            end = min(i + len(seq), upper)
            for j in range(i, end):
                weight[0, j] = w
            i = end
            current_span = name
            current_close_seq = span_ids[name]["close_seq"]
            continue

        # 3. Inside a span but not at boundary: apply current weight.
        if current_span is not None:
            weight[0, i] = span_ids[current_span]["weight"]
        # else: leave default 1.0 (between spans).
        i += 1

    return weight


def _extract_eval_positions(
    input_ids_flat: List[int],
    ans_start: int,
    ans_end: int,
    span_ids: Dict[str, Dict],
    seq_len: int,
    sample_type: str,
) -> Dict:
    """Locate token positions used by trainer.evaluate() for accuracy metrics.

    Returns:
        {
            "pre_action_position": int | None
                # v11.4: position whose ARGMAX should equal <action>.
                # The token at this position is the one BEFORE <action>
                # in the assistant span (typically </think>). Teacher-
                # forcing argmax here measures CLOSED-BOOK format
                # compliance — does the model decide to emit <action>
                # when it should? Critical: action_keyword_positions
                # measures "given <action>, predict the keyword" which
                # is open-book and missed the v11.3 cold-start bug
                # (model's <action> embedding stayed near init magnitude
                # so model never emitted <action> in free generation
                # despite scoring 0.98 on the open-book metric).
            "action_open_token_id": int | None
                # The expected token id at pre_action_position+1; trainer
                # compares argmax(logits[pre_action_position]) against this.
            "action_keyword_positions": [int, ...]
                # tokens between <action> and </action>; argmax match here
                # gives eval/action_accuracy (open-book).
            "post_action_position": int | None
                # first token after </action>; for silent samples, this
                # should equal <|im_end|>. argmax match gives
                # eval/silent_eos_rate (filtered to silent samples).
            "summary_span_positions": [int, ...]
                # tokens between <summary> and </summary> (compress samples
                # only).
            "query_span_positions": [int, ...]
                # tokens between <query> and </query> (recall_query samples).
            "response_span_positions": [int, ...]
                # v11.4: tokens between <response> and </response> (response
                # / recall_response samples). Gives eval/response_argmax_acc
                # — token-level answer-content accuracy under teacher forcing.
            "sample_type": str
                # passed through so trainer can bucket per class.
        }
    """
    meta: Dict = {
        "pre_action_position": None,
        "action_open_token_id": None,
        "action_keyword_positions": [],
        "post_action_position": None,
        "summary_span_positions": [],
        "query_span_positions": [],
        "response_span_positions": [],
        "sample_type": sample_type,
    }
    if "action" not in span_ids:
        return meta

    # v11.4: span open/close are SEQUENCES (length 1 in legacy single-token
    # mode, length 2-4 in industry multi-token mode). Walk with sequence
    # matching so the same code handles both modes.
    action_open_seq = span_ids["action"]["open_seq"]
    action_close_seq = span_ids["action"]["close_seq"]
    summary_open_seq = span_ids.get("summary", {}).get("open_seq")
    summary_close_seq = span_ids.get("summary", {}).get("close_seq")
    query_open_seq = span_ids.get("query", {}).get("open_seq")
    query_close_seq = span_ids.get("query", {}).get("close_seq")
    response_open_seq = span_ids.get("response", {}).get("open_seq")
    response_close_seq = span_ids.get("response", {}).get("close_seq")

    in_action = False
    in_summary = False
    in_query = False
    in_response = False
    saw_action_close = False
    upper = min(ans_end + 2, seq_len)
    i = ans_start
    while i < upper:
        # Action open
        if not in_action and _seq_match_at(input_ids_flat, i, action_open_seq):
            # v11.4: pre_action_position is the position right BEFORE the
            # FIRST token of the <action> sequence — the position whose
            # argmax should be the <action> sequence's first token.
            if i > 0:
                meta["pre_action_position"] = i - 1
                meta["action_open_token_id"] = action_open_seq[0]
            in_action = True
            i += len(action_open_seq)
            continue
        # Action close
        if in_action and _seq_match_at(input_ids_flat, i, action_close_seq):
            in_action = False
            saw_action_close = True
            after = i + len(action_close_seq)
            if after < seq_len:
                meta["post_action_position"] = after
            i = after
            continue
        # Action keyword tokens (between open and close)
        if in_action:
            meta["action_keyword_positions"].append(i)
            i += 1
            continue
        if not saw_action_close:
            i += 1
            continue
        # Summary span
        if summary_open_seq and not in_summary and \
                _seq_match_at(input_ids_flat, i, summary_open_seq):
            in_summary = True
            i += len(summary_open_seq)
            continue
        if summary_close_seq and in_summary and \
                _seq_match_at(input_ids_flat, i, summary_close_seq):
            in_summary = False
            i += len(summary_close_seq)
            continue
        # Query span
        if query_open_seq and not in_query and \
                _seq_match_at(input_ids_flat, i, query_open_seq):
            in_query = True
            i += len(query_open_seq)
            continue
        if query_close_seq and in_query and \
                _seq_match_at(input_ids_flat, i, query_close_seq):
            in_query = False
            i += len(query_close_seq)
            continue
        # Response span
        if response_open_seq and not in_response and \
                _seq_match_at(input_ids_flat, i, response_open_seq):
            in_response = True
            i += len(response_open_seq)
            continue
        if response_close_seq and in_response and \
                _seq_match_at(input_ids_flat, i, response_close_seq):
            in_response = False
            i += len(response_close_seq)
            continue
        # Inside a payload span: tag this token for argmax tracking.
        if in_summary:
            meta["summary_span_positions"].append(i)
        elif in_query:
            meta["query_span_positions"].append(i)
        elif in_response:
            meta["response_span_positions"].append(i)
        i += 1

    return meta


def preprocess_per_timestep(
    sample: Dict, processor, protocol_version: str = "v11"
) -> Dict:
    """Tokenize a per-timestep sample and mask labels.

    Only the assistant turn (output) contributes to loss.
    Uses processor.apply_chat_template for unified tokenization + vision.

    Accepts two formats:
    - Messages format (pipeline v5): sample["messages"] used directly
    - Legacy format: sample["input"]/["output"] built via build_per_timestep_messages()

    protocol_version controls which message builder + chat_template tools
    are used. v12 mode passes tools=TOOLS_SCHEMA to apply_chat_template so
    the system prompt auto-renders the <tools> block per Qwen2.5-VL spec.
    """
    base_path = Path(sample.get("data_path", "."))

    if "messages" in sample:
        # Pipeline v5: messages already constructed, resolve paths and use directly
        messages = _resolve_video_paths(sample["messages"], base_path)
    else:
        # Legacy: build messages from input/output structure
        if protocol_version == "v12":
            messages = build_per_timestep_messages_v12(sample, base_path)
        else:
            messages = build_per_timestep_messages(sample, base_path)

    # Tokenize with vision processing.
    # v12 mode: pass tools=TOOLS_SCHEMA so chat_template auto-renders the
    # <tools>...</tools> JSON block in the system prompt — matches the
    # DeepEyesV2 SFT pattern (multiturn_sft_dataset.py:151-157).
    template_kwargs = dict(tokenize=True, return_dict=True, return_tensors="pt")
    if protocol_version == "v12":
        from thinkstream.data.agent_protocol import TOOLS_SCHEMA
        template_kwargs["tools"] = TOOLS_SCHEMA

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

    # v11: exactly one assistant turn per sample.
    # v12: allow 1 (silent/response/compress) or 2 (multi-turn recall) turns.
    # Either way, every assistant turn contributes loss=1 (per DeepEyesV2
    # multiturn_sft_dataset.py:170 pattern: gen-prompt prefix masked,
    # message body trained).
    expected = (
        {1, 2} if protocol_version == "v12" else {1}
    )
    if len(assistant_spans) not in expected:
        sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
        raise ValueError(
            f"Sample {sid}: expected {sorted(expected)} assistant turn(s), "
            f"found {len(assistant_spans)}. v11 = 1 turn always; v12 = 1 turn "
            f"for silent/response/compress and 2 turns for recall multi-turn."
        )

    # Unmask all assistant turns (each can have its own [start, end] range).
    for ans_start, ans_end in assistant_spans:
        labels[0, ans_start: ans_end + 2] = input_ids[0, ans_start: ans_end + 2]
    # Use the FIRST span's bounds for downstream weight construction —
    # token_loss_weight in v12 is uniform 1.0 anyway, and v11 has only
    # one span so this is unchanged.
    ans_start, ans_end = assistant_spans[0]

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    if protocol_version == "v12":
        # v12: vanilla CE per DeepEyesV2 multiturn_sft_dataset.py:170 —
        # uniform weight 1.0 across the assistant span. No SPAN_WEIGHTS,
        # no ACTION_WEIGHTS, no focal+α. Imbalance handled by sampler.
        # token_loss_weight kept as a tensor of 1s so trainer.compute_loss
        # can treat it uniformly without a separate v12 code path.
        full_result["token_loss_weight"] = torch.ones(
            (1, L), dtype=torch.float
        )
        # Per-sample weight: 1.0 in v12. sample-type-aware weighting is
        # delegated entirely to ClassBalancedDistributedSampler.
        full_result["sample_weight"] = 1.0
        # eval_meta in v12: only the assistant-span boundary is meaningful;
        # action_keyword_positions are deprecated (no action vocab). We keep
        # the field for trainer-side compatibility (empty positions →
        # _accumulate_train_metrics short-circuits).
        full_result["eval_meta"] = {
            "pre_action_position": None,
            "action_open_token_id": None,
            "action_keyword_positions": [],
            "post_action_position": None,
            "summary_span_positions": [],
            "query_span_positions": [],
            "response_span_positions": [],
            "sample_type": sample.get("sample_type", "?"),
            "ans_start": ans_start,
            "ans_end": ans_end,
        }
        return full_result

    # ── v11 legacy path below ───────────────────────────────────────────
    # Per-token loss weight: <think> low, <action> high, <response>/<query>/<summary> medium.
    # See SPAN_WEIGHTS docstring and trainer.py:compute_loss for usage.
    span_ids = _resolve_span_token_ids(processor.tokenizer)
    full_result["token_loss_weight"] = _build_token_loss_weight(
        input_ids_flat, ans_start, ans_end, span_ids, L,
    )

    # Per-sample loss weight (context-aware, not just sample_type)
    full_result["sample_weight"] = _get_sample_weight(sample)

    # Eval-time accuracy probes: positions of action keyword token(s) +
    # the post-action transition token. Used by trainer.evaluate() to
    # compute eval/action_accuracy and eval/silent_eos_rate without
    # running generation. Inference time = teacher-forced argmax match.
    full_result["eval_meta"] = _extract_eval_positions(
        input_ids_flat, ans_start, ans_end, span_ids, L,
        sample_type=sample.get("sample_type", "?"),
    )

    return full_result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PerTimestepDataset(Dataset):
    """Dataset for per-timestep agent SFT.

    Each sample is a single 2s chunk with memory state, visual window,
    optional recalled frames, and a single assistant output.
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

        # v11.3: defensive empty-sample filter. Catches:
        #  (a) the 8 zero-chunk videos in batch1 that pass2 failed on —
        #      they shouldn't produce SFT samples (pass3c needs thinks),
        #      but if a corrupted row slips through the filter is cheap;
        #  (b) any sample missing input.system / input.visual_window /
        #      output (schema corruption);
        #  (c) samples whose visual_window has 0 frames (pass2 rollout
        #      truncated at chunk 0 with no useful content).
        before = len(all_samples)
        all_samples = [
            s for s in all_samples
            if isinstance(s.get("input"), dict)
            and s["input"].get("system")
            and isinstance(s["input"].get("visual_window"), dict)
            and s["input"]["visual_window"].get("frames", 0) > 0
            and s.get("output")
        ]
        empty_dropped = before - len(all_samples)
        if empty_dropped > 0:
            rank0_print(f"  Dropped {empty_dropped} empty/corrupted samples "
                        f"(missing input/visual_window/output)")

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
        # v12.0: protocol selector — propagated to preprocess_per_timestep
        self.protocol_version = getattr(data_args, "protocol_version", "v11")
        if self.protocol_version not in ("v11", "v12"):
            raise ValueError(
                f"protocol_version must be 'v11' or 'v12', "
                f"got {self.protocol_version!r}"
            )
        rank0_print(f"PerTimestepDataset protocol_version={self.protocol_version}")

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

        # Tokenize + vision + label mask. protocol_version drives chat_template
        # tools= injection and disables v11 SPAN_WEIGHTS / focal+α machinery.
        data_dict = preprocess_per_timestep(
            sample, self.processor, protocol_version=self.protocol_version,
        )

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

        # Per-sample loss weights
        batch["sample_weights"] = torch.tensor(
            [inst.get("sample_weight", 1.0) for inst in instances],
            dtype=torch.float32,
        )

        # Per-sample metadata (audit logging only; popped before model.forward)
        batch["sample_meta"] = [inst.get("sample_meta", {}) for inst in instances]

        # Eval-time accuracy probes (popped before model.forward in trainer)
        batch["eval_meta"] = [inst.get("eval_meta", {}) for inst in instances]

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
        extra_weights: Optional[List[float]] = None,
    ):
        """
        Args:
            extra_weights: optional per-sample multiplier (v11.3) — used for
                unique-think-rate weighting so static-scene videos with
                duplicate thinks get sampled less. Length must match
                sample_types. Multiplied into the inv-freq class weights
                AFTER class normalization, so the class-balance property
                is preserved on average.
        """
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

        weights_list = [cls_weights[t] for t in sample_types]
        if extra_weights is not None:
            if len(extra_weights) != len(sample_types):
                raise ValueError(
                    f"extra_weights length {len(extra_weights)} != "
                    f"sample_types length {len(sample_types)}"
                )
            weights_list = [w * float(e) for w, e in zip(weights_list, extra_weights)]
        self.weights = torch.tensor(weights_list, dtype=torch.double)

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

    # Build eval dataset if requested
    eval_dataset = None
    eval_names = getattr(data_args, "eval_dataset_use", "")
    if eval_names:
        from dataclasses import replace
        eval_args = replace(data_args, dataset_use=eval_names)
        eval_dataset = PerTimestepDataset(processor, eval_args)
        rank0_print(f"Eval samples: {len(eval_dataset)}")

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
