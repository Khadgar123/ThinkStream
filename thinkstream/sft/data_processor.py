"""Per-timestep agent SFT data processor.

Based on Qwen3-VL official finetune data processing, adapted to the
ShareGPT messages format emitted by pass5. Each sample
is one inference-step snapshot with assistant-span CE labels only.

Key differences from standard VLM SFT:
- Input is pre-rendered ShareGPT messages, not ad-hoc flat JSON
- Messages contain <memory>, <visual_window>, <recalled_frames> tags
- Labels mask prompt/tool/user tokens and train only assistant spans
- Samples are independent per-timestep snapshots

See docs/sft_engineering.md §2 and docs/data_construction_zh.md §13.
"""

import json
import os
import random
import logging
import time
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Any, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from .data_list import data_list
from thinkstream.data.rope2d import get_rope_index_25, get_rope_index_3

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
      (1) Messages format (post-pass5): sum text in content + count visual frames
      (2) Flat format: parse input.{system,memory,queries,visual_window} fields

    Vision token cost per frame tracks the runtime 130k-220k pixel profile
    (2 fps, merge_size=2). The estimate is intentionally conservative; it
    only needs to rank samples for length grouping and overlong filtering.
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
                    elif item.get("image") or item.get("image_url") or t == "image":
                        n_frames += 1
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


def _sample_has_visual(sample: Dict) -> bool:
    """Whether this row will execute the vision path in model.forward."""
    if "messages" in sample:
        for msg in sample.get("messages") or []:
            content = msg.get("content")
            parts = content if isinstance(content, list) else []
            for item in parts:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in ("image", "video"):
                    return True
                if item.get("image") or item.get("image_url") or item.get("video"):
                    return True
        return False

    inp = sample.get("input", {}) or {}
    vw = inp.get("visual_window") or {}
    if int(vw.get("frames", 0) or 0) > 0:
        return True
    rf = inp.get("recalled_frames") or {}
    return int(rf.get("n_frames", 0) or 0) > 0


def _parse_ratio_spec(spec: Optional[str]) -> Dict[str, float]:
    if not spec:
        return {}
    ratios: Dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid class_loss_target_ratios item: {part!r}")
        key, value = part.split("=", 1)
        ratios[key.strip()] = float(value)
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("class_loss_target_ratios must sum to a positive value")
    return {k: v / total for k, v in ratios.items()}


def _sample_loss_class(sample: Dict) -> str:
    """Class used for SFT loss weighting and train-time diagnostics.

    ``sample_type`` stays protocol-facing. After pass5 recall splitting,
    ``sample_type=recall`` covers both the recall tool-call row and the
    post-recall answer row, which are different training behaviours. Use
    ``sft_subtype`` to keep active recall weighting/metrics from being diluted
    by the second no-tools answer turn.
    """
    meta = sample.get("metadata") or {}
    explicit = (
        sample.get("loss_class")
        or sample.get("sft_loss_class")
        or meta.get("loss_class")
        or meta.get("sft_loss_class")
    )
    if explicit:
        return str(explicit)

    subtype = str(sample.get("sft_subtype") or meta.get("sft_subtype") or "").strip().lower()
    if "recall_query" in subtype:
        return "recall"
    if "post_recall" in subtype or subtype in {"recall_answer", "recall_response"}:
        return "post_recall"

    stype = str(sample.get("sample_type") or "?")
    if stype in ("recall_response", "post_recall"):
        return "post_recall"
    return stype


def _assign_class_loss_weights(samples: List[Dict], data_args) -> None:
    """Assign normalized class weights after filtering, without resampling."""
    ratios = _parse_ratio_spec(getattr(data_args, "class_loss_target_ratios", None))
    alpha = float(getattr(data_args, "class_loss_alpha", 1.0) or 0.0)
    if not ratios or alpha <= 0 or not samples:
        for s in samples:
            s["_loss_class"] = _sample_loss_class(s)
            s["_sample_weight"] = 1.0
        return

    for s in samples:
        s["_loss_class"] = _sample_loss_class(s)

    counts = Counter(s.get("_loss_class", "?") for s in samples)
    total = sum(counts.values())
    weights: Dict[str, float] = {}
    max_weight = float(getattr(data_args, "class_loss_max_weight", 8.0) or 0.0)
    for stype, n in counts.items():
        observed = n / total
        target = ratios.get(stype, observed)
        w = (target / observed) ** alpha if observed > 0 else 1.0
        if max_weight > 0:
            w = min(w, max_weight)
        weights[stype] = w

    mean_w = sum((counts[k] / total) * weights[k] for k in counts)
    if mean_w <= 0:
        mean_w = 1.0
    for k in list(weights):
        weights[k] /= mean_w
    for s in samples:
        s["_sample_weight"] = float(weights.get(s.get("_loss_class", "?"), 1.0))

    rank0_print(
        "Class loss weights:",
        {k: round(weights[k], 4) for k in sorted(weights)},
        "observed:",
        {k: round(counts[k] / total, 4) for k in sorted(counts)},
        "target:",
        {k: round(ratios.get(k, counts[k] / total), 4) for k in sorted(counts)},
    )


def _sample_rank_for_eval(sample: Dict, idx: int, seed: int = 0) -> str:
    key = "|".join([
        str(seed),
        str(sample.get("video_id") or (sample.get("metadata") or {}).get("video_id") or ""),
        str(sample.get("trajectory_id") or ""),
        str(sample.get("sample_id") or ""),
        str(sample.get("chunk_idx") or ""),
        str(idx),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _message_sample_type(sample: Dict) -> str:
    meta = sample.get("metadata") or {}
    return str(sample.get("sample_type") or meta.get("sample_type") or "")


def _message_chunk_idx(sample: Dict) -> Optional[int]:
    meta = sample.get("metadata") or {}
    for value in (sample.get("chunk_idx"), meta.get("chunk_idx")):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_list(value: Any) -> List[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    out: List[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _answer_chunks_from_metadata(meta: Dict[str, Any]) -> List[int]:
    chunks: List[int] = []
    chunks.extend(_int_list(meta.get("answer_chunks")))
    chunks.extend(_int_list(meta.get("expected_answer_chunks")))
    for item in meta.get("per_emit_answers") or []:
        if not isinstance(item, dict):
            continue
        try:
            chunks.append(int(item.get("chunk")))
        except (TypeError, ValueError):
            continue
    return sorted(set(c for c in chunks if c >= 0))


def _eval_silent_role(sample: Dict) -> str:
    """Pass5-style silent subtype for eval balancing.

    Rendered message rows no longer carry the full pass3 query objects, so this
    mirrors pass5 when possible and falls back to metadata timing fields.
    """
    if _sample_loss_class(sample) != "silent" and _message_sample_type(sample) != "silent":
        return ""

    meta = sample.get("metadata") or {}
    queries = list(sample.get("queries") or meta.get("queries") or [])
    card_id = str(sample.get("card_id") or meta.get("card_id") or "")
    if queries:
        related = [
            q for q in queries
            if not card_id or str(q.get("card_id") or "") == card_id
        ] or queries
        if any(str(q.get("status", "")).lower() in {"open", "pending", "active"}
               for q in related):
            return "pending_question"
        if any(q.get("answers") for q in related):
            return "post_answer"

    if not (card_id or meta.get("question")):
        return "no_question"

    chunk_idx = _message_chunk_idx(sample)
    answer_chunks = _answer_chunks_from_metadata(meta)
    if chunk_idx is not None and answer_chunks:
        if any(c >= chunk_idx for c in answer_chunks):
            return "pending_question"
        return "post_answer"
    return "pending_question"


def _eval_pending_silent_temporal_bucket(sample: Dict) -> str:
    if _eval_silent_role(sample) != "pending_question":
        return "none"
    meta = sample.get("metadata") or {}
    chunk_idx = _message_chunk_idx(sample)
    if chunk_idx is None:
        return "unknown"
    try:
        ask_chunk = int(meta.get("ask_chunk"))
    except (TypeError, ValueError):
        ask_chunk = None

    future_answers = [
        c for c in _answer_chunks_from_metadata(meta)
        if c >= chunk_idx
    ]
    next_answer = min(future_answers) if future_answers else None
    distance_to_answer = (
        next_answer - chunk_idx if next_answer is not None else None
    )

    if ask_chunk is not None:
        distance_from_ask = chunk_idx - ask_chunk
        if 0 <= distance_from_ask <= 1:
            return "ask_edge"
    if distance_to_answer is not None and 0 <= distance_to_answer <= 1:
        return "answer_edge"
    if ask_chunk is not None and 2 <= chunk_idx - ask_chunk <= 3:
        return "near_ask"
    if distance_to_answer is not None and 2 <= distance_to_answer <= 3:
        return "near_answer"
    if ask_chunk is None and distance_to_answer is None:
        return "unknown"
    return "middle"


def _eval_silent_diversity_weight(sample: Dict) -> int:
    bucket = _eval_pending_silent_temporal_bucket(sample)
    if bucket in {"ask_edge", "answer_edge"}:
        return 4
    if bucket in {"near_ask", "near_answer"}:
        return 2
    return 1


def _eval_silent_diversity_key(sample: Dict) -> str:
    role = _eval_silent_role(sample)
    meta = sample.get("metadata") or {}
    family = str(meta.get("family") or "none")
    answer_form = str(meta.get("answer_form") or "none")
    availability = str(
        meta.get("availability")
        or sample.get("sequence_type")
        or "none"
    )
    question_type = str(meta.get("question_type") or "single_emit")
    base_role = str(sample.get("base_role") or meta.get("base_role") or "")

    if role == "pending_question":
        if base_role == "recall_wait_no_history":
            subtype = "recall_wait_no_history"
        elif availability == "event_watch":
            subtype = "future_event_wait"
        elif availability == "multi_response" or question_type == "multi_emit":
            subtype = "multi_emit_wait"
        elif availability == "recall_success":
            subtype = "recall_answer_pending"
        elif availability == "memory_response":
            subtype = "memory_answer_pending"
        elif availability == "immediate_response":
            subtype = "immediate_boundary_wait"
        else:
            subtype = availability or "pending"
        temporal = _eval_pending_silent_temporal_bucket(sample)
        return f"{role}|{subtype}|{family}|{answer_form}|{question_type}|{temporal}"

    if role == "post_answer":
        return f"{role}|{family}|{answer_form}|{question_type}"
    return f"{role or 'unknown'}|{base_role or 'patrol'}"


def _choose_ranked_eval(
    items: List[Tuple[int, Dict]],
    n: int,
    *,
    seed: int,
) -> List[Tuple[int, Dict]]:
    if n <= 0:
        return []
    return sorted(items, key=lambda x: _sample_rank_for_eval(x[1], x[0], seed))[:n]


def _choose_diverse_eval_silent(
    items: List[Tuple[int, Dict]],
    n: int,
    *,
    seed: int,
) -> List[Tuple[int, Dict]]:
    if n <= 0 or not items:
        return []
    by_key: Dict[str, List[Tuple[int, Dict]]] = {}
    for item in items:
        by_key.setdefault(_eval_silent_diversity_key(item[1]), []).append(item)
    for key in by_key:
        by_key[key] = _choose_ranked_eval(by_key[key], len(by_key[key]), seed=seed)
    key_weights = {
        key: max(_eval_silent_diversity_weight(sample) for _idx, sample in bucket)
        for key, bucket in by_key.items()
    }

    selected: List[Tuple[int, Dict]] = []
    cursors = {key: 0 for key in by_key}
    keys = sorted(by_key, key=lambda k: (-key_weights[k], -len(by_key[k]), k))
    while len(selected) < n:
        progressed = False
        for key in keys:
            for _ in range(max(1, key_weights[key])):
                cur = cursors[key]
                bucket = by_key[key]
                if cur >= len(bucket):
                    break
                selected.append(bucket[cur])
                cursors[key] += 1
                progressed = True
                if len(selected) >= n:
                    break
            if len(selected) >= n:
                break
        if not progressed:
            break
    return selected


def _choose_eval_pending_silent(
    items: List[Tuple[int, Dict]],
    n: int,
    *,
    seed: int,
) -> List[Tuple[int, Dict]]:
    if n <= 0 or not items:
        return []
    temporal_floors = {
        "ask_edge": 0.32,
        "answer_edge": 0.25,
        "near_ask": 0.08,
        "near_answer": 0.12,
    }
    by_temporal: Dict[str, List[Tuple[int, Dict]]] = {}
    for item in items:
        by_temporal.setdefault(
            _eval_pending_silent_temporal_bucket(item[1]),
            [],
        ).append(item)

    selected: List[Tuple[int, Dict]] = []
    used: set[int] = set()
    for bucket, ratio in temporal_floors.items():
        bucket_items = by_temporal.get(bucket, [])
        target = min(len(bucket_items), int(n * ratio))
        if target <= 0:
            continue
        picked = _choose_diverse_eval_silent(bucket_items, target, seed=seed)
        selected.extend(picked)
        used.update(i for i, _s in picked)

    if len(selected) < n:
        remaining = [(i, s) for i, s in items if i not in used]
        selected.extend(
            _choose_diverse_eval_silent(remaining, n - len(selected), seed=seed)
        )
    return selected[:n]


def _choose_eval_silent(
    items: List[Tuple[int, Dict]],
    n: int,
    *,
    seed: int,
    diverse: bool,
) -> List[Tuple[int, Dict]]:
    if n <= 0 or not items:
        return []
    if not diverse:
        return _choose_ranked_eval(items, n, seed=seed)

    pending = [(i, s) for i, s in items if _eval_silent_role(s) == "pending_question"]
    post_answer = [(i, s) for i, s in items if _eval_silent_role(s) == "post_answer"]
    base = [(i, s) for i, s in items if _eval_silent_role(s) == "no_question"]
    target_pending = min(len(pending), int(n * 0.55))
    target_post = min(len(post_answer), int(n * 0.25))
    selected = (
        _choose_eval_pending_silent(pending, target_pending, seed=seed)
        + _choose_diverse_eval_silent(post_answer, target_post, seed=seed)
    )
    used = {i for i, _s in selected}
    remaining_n = n - len(selected)
    if remaining_n > 0:
        selected.extend(_choose_diverse_eval_silent(base, remaining_n, seed=seed))
        used = {i for i, _s in selected}
    if len(selected) < n:
        rest = [
            (i, s) for bucket in (pending, post_answer, base)
            for i, s in bucket
            if i not in used
        ]
        selected.extend(_choose_diverse_eval_silent(rest, n - len(selected), seed=seed))
    return selected[:n]


def _allocate_eval_quotas(
    counts: Counter,
    total: int,
    ratios: Dict[str, float],
) -> Dict[str, int]:
    classes = [k for k, n in counts.items() if n > 0]
    if not classes or total <= 0:
        return {}
    if not ratios:
        ratios = {k: 1.0 / len(classes) for k in classes}
    else:
        # Keep unspecified present classes from being silently excluded.
        missing = [k for k in classes if k not in ratios]
        if missing:
            leftover = max(0.0, 1.0 - sum(ratios.values()))
            share = leftover / len(missing) if leftover > 0 else 0.0
            ratios = {**ratios, **{k: share for k in missing}}
    desired = {k: total * float(ratios.get(k, 0.0)) for k in classes}
    quotas = {k: min(counts[k], int(desired[k])) for k in classes}
    remaining = min(total, sum(counts.values())) - sum(quotas.values())
    while remaining > 0:
        candidates = [
            k for k in classes
            if quotas[k] < counts[k]
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda k: (
                desired.get(k, 0.0) - quotas[k],
                counts[k] - quotas[k],
                k,
            ),
            reverse=True,
        )
        quotas[candidates[0]] += 1
        remaining -= 1
    return quotas


def _subsample_eval_balanced(
    samples: List[Dict],
    max_samples: int,
    data_args,
) -> List[Dict]:
    strategy = str(getattr(data_args, "eval_balance_strategy", "none") or "none").lower()
    if strategy in {"", "none", "random"}:
        rng = random.Random(int(getattr(data_args, "eval_balance_seed", 0) or 0))
        return rng.sample(samples, max_samples)
    if strategy not in {"loss_class", "loss_class_silent_diverse"}:
        raise ValueError(
            "eval_balance_strategy must be one of: none, random, "
            "loss_class, loss_class_silent_diverse"
        )

    seed = int(getattr(data_args, "eval_balance_seed", 0) or 0)
    by_class: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, sample in enumerate(samples):
        by_class.setdefault(_sample_loss_class(sample), []).append((idx, sample))
    counts = Counter({k: len(v) for k, v in by_class.items()})
    ratios = _parse_ratio_spec(getattr(data_args, "eval_balance_target_ratios", None))
    quotas = _allocate_eval_quotas(counts, max_samples, ratios)

    selected: List[Tuple[int, Dict]] = []
    for cls in sorted(quotas):
        items = by_class.get(cls, [])
        if cls == "silent":
            picked = _choose_eval_silent(
                items,
                quotas[cls],
                seed=seed,
                diverse=(strategy == "loss_class_silent_diverse"),
            )
        else:
            picked = _choose_ranked_eval(items, quotas[cls], seed=seed)
        selected.extend(picked)
    selected.sort(key=lambda x: x[0])
    out = [sample for _idx, sample in selected]

    silent_roles = Counter(_eval_silent_role(s) for s in out if _sample_loss_class(s) == "silent")
    pending_temporal = Counter(
        _eval_pending_silent_temporal_bucket(s)
        for s in out
        if _sample_loss_class(s) == "silent" and _eval_silent_role(s) == "pending_question"
    )
    rank0_print(
        f"  Balanced eval set to {len(out)} using strategy={strategy}",
        "class_counts:",
        dict(sorted(Counter(_sample_loss_class(s) for s in out).items())),
        "source_counts:",
        dict(sorted(counts.items())),
        "silent_roles:",
        dict(sorted((k or 'unknown', v) for k, v in silent_roles.items())),
        "pending_temporal:",
        dict(sorted(pending_temporal.items())),
    )
    return out


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
from thinkstream.data.agent_protocol import (
    format_memory_block as _shared_format_memory,
    format_user_input_block,
)


def _format_memory_block(memory: Dict) -> str:
    """Format memory state as text. Delegates to shared agent_protocol."""
    return _shared_format_memory(memory)


def _resolve_frame_paths(paths: List[str], base_path: Path) -> List[str]:
    """Resolve frame paths after a batch directory is copied to a new root."""
    roots = []
    for value in (os.environ.get("THINKSTREAM_DATA_ROOT"), os.environ.get("AGENT_DATA_DIR")):
        if value:
            root = Path(value)
            if root not in roots:
                roots.append(root)
    if base_path not in roots:
        roots.append(base_path)
    out: List[str] = []
    for raw in paths:
        p = Path(str(raw))
        parts = p.parts
        if "frames" in parts:
            idx = parts.index("frames")
            for root in roots:
                candidate = root / "frames" / Path(*parts[idx + 1:])
                if candidate.exists():
                    out.append(str(candidate))
                    break
            else:
                out.append(str(p if p.is_absolute() else base_path / p))
            continue
        if not p.is_absolute():
            direct = base_path / p
            if direct.exists():
                out.append(str(direct))
                continue
            out.append(str(direct))
            continue
        if p.exists():
            out.append(str(p))
            continue
        out.append(str(p))
    return out


def build_per_timestep_messages(sample: Dict, base_path: Path) -> List[Dict]:
    """v12.0: Build messages for the official Qwen tool-call protocol.

    DEPRECATED: the canonical builder is now
    ``scripts/agent_data/pass5_messages.py:build_messages``, which is used
    by the main pipeline (`pass5_messages.py:convert`). This function is kept
    only for legacy eval/debug paths and mirrors the current pass5 contract as
    closely as possible.

    Three sample shapes handled (controlled by pass3c-emitted fields):

    A. Single-turn (silent / response / lonely recall):
       sample["output"] = single assistant string. Messages = [system, user, assistant].

    B. Multi-turn recall (sample_type=='recall' with v12_assistant_turn_1/2):
       Two assistant turns sandwiching a tool turn. Messages =
         [system, user (chunk visual+memory+query),
          assistant (tool_call recall),
          user (recalled_frames + metadata-only recall_result),
          assistant (final answer)]
       This implements the within-one-chunk agentic cycle (think→recall→
       result→think→answer) per docs/v12.0_protocol_migration_design.md §1.

    C. Inter-chunk compress (inter_chunk=True):
       The user_input compress trigger is rendered before memory, and the
       prompt omits visual_window/images/videos because compression is a
       text-memory action between visual timesteps.

    Differences from v11 (build_per_timestep_messages):
    - SYSTEM_PROMPT (concise; <tools> block rendered by chat_template
      via tools= parameter at apply time).
    - recall_result is metadata-only; historical frames carry recall evidence.
      Shape-B recall uses a dedicated 'tool' role message (matches Qwen3-VL chat_template tool branch which
      nests <tool_response> inside the <|im_start|>user wrapper).
    """
    # v12.6: import canonical chunk_sec via agent_protocol (which already
    # falls back gracefully when scripts.agent_data.config isn't on the
    # path — e.g. inference container). Earlier the direct
    # `from scripts.agent_data.config import AGENT_CHUNK_SEC` would raise
    # ModuleNotFoundError when training was launched outside the project
    # root. Going through agent_protocol routes through the same fallback
    # chain SFT/eval/RL all use.
    from thinkstream.data.agent_protocol import (
        AGENT_CHUNK_SEC,
        FRAMES_PER_CHUNK,
        append_visual_frames,
        build_recall_result_metadata,
        format_queries_block,
        is_inter_chunk,
        normalize_frame_protocol,
        prompt_time_range,
        prompt_time_value,
        system_prompt_for_frame_protocol,
    )

    inp = sample["input"]
    chunk_idx = sample["chunk_idx"]
    chunk_sec = float(AGENT_CHUNK_SEC)
    frame_protocol = normalize_frame_protocol(sample.get("frame_protocol"))
    inter_chunk = is_inter_chunk(sample)
    is_recall_multiturn = (
        sample.get("sample_type") == "recall"
        and "v12_assistant_turn_1" in sample
    )
    sample_type = str(sample.get("sample_type") or "").strip().lower()
    explicit_post_recall = sample_type in {
        "post_recall",
        "recall_response",
        "recall_answer",
    }
    legacy_post_recall = (
        explicit_post_recall
        or (bool(inp.get("recall_result")) and not is_recall_multiturn and sample_type != "recall")
    )

    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt_for_frame_protocol(
                    frame_protocol,
                    prompt_kind=(
                        "post_recall"
                        if (
                            legacy_post_recall
                        )
                        else None
                    ),
                    inter_chunk=inter_chunk,
                ),
            }],
        }
    ]

    video_path = sample.get("video_path", "")
    if video_path and not Path(video_path).is_absolute():
        video_path = str(base_path / video_path)
    require_pre = bool(sample.get("_require_pre_extracted_frames", True))

    # ── User content ───────────────────────────────────────────────────
    user_content = []

    if legacy_post_recall and not is_recall_multiturn and not inter_chunk:
        queries_text = format_queries_block(inp.get("queries", []))
        if queries_text:
            user_content.append({"type": "text", "text": queries_text})

        rf = inp.get("recalled_frames") or {}
        if rf:
            rf_header = json.dumps({
                "time_range": prompt_time_range(rf["time_range"]),
                "source": rf.get("source", "historical_frames"),
                "n_frames": rf["n_frames"],
            })
            user_content.append({
                "type": "text",
                "text": f"\n<recalled_frames>{rf_header}</recalled_frames>",
            })
            if "frame_paths" in rf:
                paths = _resolve_frame_paths(rf["frame_paths"], base_path)
                try:
                    from scripts.agent_data.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
                start_frame = int(round(float(rf["time_range"][0]) / chunk_sec)) * FRAMES_PER_CHUNK
                total_frames = int(round(float(rf["time_range"][1]) / chunk_sec)) * FRAMES_PER_CHUNK
                append_visual_frames(
                    user_content,
                    paths,
                    frame_protocol=frame_protocol,
                    fps=float(FRAMES_PER_CHUNK / chunk_sec),
                    start_frame_index=start_frame,
                    total_num_frames=total_frames,
                    context_label="recalled frame",
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )
            elif video_path and not require_pre:
                user_content.append({
                    "type": "video", "video": video_path,
                    "video_start": prompt_time_value(rf["time_range"][0]),
                    "video_end": prompt_time_value(rf["time_range"][1]),
                })

        rr_json = json.dumps(
            build_recall_result_metadata(inp.get("recall_result") or {}, rf),
            ensure_ascii=False,
        )
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample.get("output", "")}],
        })
        return messages

    user_input_block = format_user_input_block(
        inp.get("user_input", ""),
        inter_chunk=inter_chunk,
    )
    if user_input_block:
        user_content.append({
            "type": "text",
            "text": user_input_block.lstrip("\n"),
        })

    # Memory follows the fresh user event so questions/triggers are visible
    # before long historical text.
    memory_text = _format_memory_block(inp["memory"])
    user_content.append({
        "type": "text",
        "text": f"\n<memory>\n{memory_text}\n</memory>" if user_content
        else f"<memory>\n{memory_text}\n</memory>",
    })

    if inter_chunk:
        messages.append({"role": "user", "content": user_content})
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample.get("output", "")}],
        })
        return messages

    # Visual window + frames.
    vw = inp["visual_window"]
    current_start = chunk_idx * chunk_sec
    current_end = current_start + chunk_sec
    vw_header = json.dumps({
        "start": vw["video_start"],
        "end": vw["video_end"],
        "frames": vw["frames"],
        "current_time": current_start,
    })
    user_content.append({
        "type": "text",
        "text": f"\n<visual_window>{vw_header}</visual_window>",
    })

    # v12.5 fallback: pass4 flat files may omit frame_paths — infer from
    # video_id + frames count using the pre-extracted frame directory.
    if "frame_paths" not in vw and "frames" in vw:
        vid = sample.get("video_id", "")
        if vid:
            try:
                from scripts.agent_data.config import (
                    DATA_ROOT as _DATA_ROOT,
                    PROJECT_ROOT as _PROJECT_ROOT,
                    VISUAL_WINDOW_CHUNKS as _VWC,
                    compute_visual_window_start as _cvws,
                )
                _frame_dir_path = _DATA_ROOT / "frames" / vid
                try:
                    frame_dir = str(_frame_dir_path.relative_to(_PROJECT_ROOT))
                except ValueError:
                    frame_dir = str(_frame_dir_path)
            except ImportError:
                data_root = (
                    os.environ.get("THINKSTREAM_DATA_ROOT")
                    or os.environ.get("AGENT_DATA_DIR")
                )
                if data_root:
                    root_path = Path(data_root)
                    frame_dir = str(
                        root_path.parent / "frames" / vid
                        if root_path.name == "final"
                        else root_path / "frames" / vid
                    )
                else:
                    frame_dir = f"data/agent_v5/frames/{vid}"
                _VWC = 16
                _cvws = lambda ck, visual_window_chunks=16: max(
                    0, int(ck) - int(visual_window_chunks) + 1
                )
            window_start = _cvws(chunk_idx, _VWC)
            paths: List[str] = []
            for ci in range(window_start, chunk_idx + 1):
                for fi in range(FRAMES_PER_CHUNK):
                    fnum = ci * FRAMES_PER_CHUNK + fi + 1
                    paths.append(f"{frame_dir}/frame_{fnum:06d}.jpg")
            vw["frame_paths"] = paths

    if "frame_paths" in vw:
        paths = _resolve_frame_paths(vw["frame_paths"], base_path)
        try:
            from scripts.agent_data.config import (
                RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
            )
        except ImportError:
            _RTKW = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
        start_frame = int(round(float(vw["video_start"]) / chunk_sec)) * FRAMES_PER_CHUNK
        total_frames = int(round(float(vw["video_end"]) / chunk_sec)) * FRAMES_PER_CHUNK
        append_visual_frames(
            user_content,
            paths,
            frame_protocol=frame_protocol,
            fps=float(FRAMES_PER_CHUNK / chunk_sec),
            start_frame_index=start_frame,
            total_num_frames=total_frames,
            latest_start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
            min_pixels=_RTKW["min_pixels"],
            max_pixels=_RTKW["max_pixels"],
        )
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

    # Active query plus response history for that same query. Query-last is the
    # canonical layout used by pass5/runtime/eval.
    queries = inp.get("queries", [])
    if queries:
        queries_text = format_queries_block(queries)
        if queries_text:
            user_content.append({"type": "text", "text": f"\n{queries_text}"})

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
            paths = _resolve_frame_paths(rf["frame_paths"], base_path)
            try:
                from scripts.agent_data.config import (
                    RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                )
            except ImportError:
                _RTKW = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
            start_frame = int(round(float(rf["time_range"][0]) / chunk_sec)) * FRAMES_PER_CHUNK
            total_frames = int(round(float(rf["time_range"][1]) / chunk_sec)) * FRAMES_PER_CHUNK
            append_visual_frames(
                user_content,
                paths,
                frame_protocol=frame_protocol,
                fps=float(FRAMES_PER_CHUNK / chunk_sec),
                start_frame_index=start_frame,
                total_num_frames=total_frames,
                context_label="recalled frame",
                min_pixels=_RTKW["min_pixels"],
                max_pixels=_RTKW["max_pixels"],
            )
        elif video_path and not require_pre:
            user_content.append({
                "type": "video", "video": video_path,
                "video_start": rf["time_range"][0],
                "video_end": rf["time_range"][1],
            })

    # Legacy (non-multi-turn) recall_result fallback. Model-visible recall
    # result is metadata only; visual evidence comes from recalled frames.
    if inp.get("recall_result") and not is_recall_multiturn and not inter_chunk:
        rr = inp["recall_result"]
        rr_json = json.dumps(
            build_recall_result_metadata(rr, inp.get("recalled_frames")),
            ensure_ascii=False,
        )
        user_content.append({
            "type": "text",
            "text": f"\n<recall_result>{rr_json}</recall_result>",
        })

    messages.append({"role": "user", "content": user_content})

    # ── Assistant turn(s) ──────────────────────────────────────────────
    if is_recall_multiturn:
        # Shape B: 2 assistant turns sandwiching a tool turn.
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": sample["v12_assistant_turn_1"]}],
        })

        # Tool turn — historical frames plus metadata-only recall_result. The
        # Qwen3-VL chat_template renders this nested under <|im_start|>user
        # but loss-masked at training time (assistant span only contributes).
        rr = sample.get("recall_result") or {}
        tool_payload = []

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
                paths = _resolve_frame_paths(rf["frame_paths"], base_path)
                try:
                    from scripts.agent_data.config import (
                        RUNTIME_MM_PROCESSOR_KWARGS as _RTKW,
                    )
                except ImportError:
                    _RTKW = {"min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28}
                start_frame = int(round(float(rf["time_range"][0]) / chunk_sec)) * FRAMES_PER_CHUNK
                total_frames = int(round(float(rf["time_range"][1]) / chunk_sec)) * FRAMES_PER_CHUNK
                append_visual_frames(
                    tool_payload,
                    paths,
                    frame_protocol=frame_protocol,
                    fps=float(FRAMES_PER_CHUNK / chunk_sec),
                    start_frame_index=start_frame,
                    total_num_frames=total_frames,
                    context_label="recalled frame",
                    min_pixels=_RTKW["min_pixels"],
                    max_pixels=_RTKW["max_pixels"],
                )
            elif video_path and not require_pre:
                tool_payload.append({
                    "type": "video", "video": video_path,
                    "video_start": rf["time_range"][0],
                    "video_end": rf["time_range"][1],
                })
        rr_json = json.dumps(
            build_recall_result_metadata(rr, rf),
            ensure_ascii=False,
        )
        tool_payload.append({
            "type": "text",
            "text": f"<recall_result>{rr_json}</recall_result>",
        })
        messages.append({"role": "user", "content": tool_payload})

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

def _resolve_media_path(value, base_path: Path):
    if isinstance(value, str) and value and not Path(value).is_absolute():
        return str(base_path / value)
    if isinstance(value, list):
        return [
            str(base_path / v)
            if isinstance(v, str) and v and not Path(v).is_absolute()
            else v
            for v in value
        ]
    return value


def _resolve_video_paths(messages: List[Dict], base_path: Path) -> List[Dict]:
    """Resolve relative media paths in messages to absolute paths."""
    resolved = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue
                if "video" in item:
                    item = dict(item)
                    item["video"] = _resolve_media_path(
                        item.get("video", ""), base_path,
                    )
                if "image" in item:
                    item = dict(item)
                    item["image"] = _resolve_media_path(
                        item.get("image", ""), base_path,
                    )
                if (
                    item.get("type") == "video"
                    and item.get("visual_carrier") == "image_pad"
                    and (item.get("image") or item.get("image_url"))
                ):
                    # Dataset routing/audits keep this as type=video, but
                    # Qwen processors only materialize image-pad frames when
                    # they are handed to apply_chat_template as type=image.
                    # Drop render-time pixel hints so the run-level
                    # --min_pixels/--max_pixels settings are the single source
                    # of truth for image-pad SFT resolution.
                    item = dict(item)
                    item["type"] = "image"
                    item.pop("visual_carrier", None)
                    item.pop("min_pixels", None)
                    item.pop("max_pixels", None)
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



def _select_loss_assistant_spans(
    assistant_spans: List[tuple],
    loss_spec,
) -> tuple[List[tuple], List[int]]:
    """Select assistant spans that should contribute loss for one row."""
    if loss_spec is None:
        loss_spec = "all"
    if isinstance(loss_spec, str):
        spec = loss_spec.strip().lower()
        if spec in ("", "all"):
            return list(assistant_spans), list(range(len(assistant_spans)))
        if spec == "last":
            return [assistant_spans[-1]], [len(assistant_spans) - 1]
        if spec == "first":
            return [assistant_spans[0]], [0]
        if spec.isdigit() or (spec.startswith("-") and spec[1:].isdigit()):
            loss_spec = [int(spec)]
        else:
            raise ValueError(f"unknown loss_assistant_turns={loss_spec!r}")
    elif isinstance(loss_spec, int):
        loss_spec = [loss_spec]

    if isinstance(loss_spec, (list, tuple)):
        out: List[tuple] = []
        indices: List[int] = []
        n = len(assistant_spans)
        for raw_idx in loss_spec:
            idx = int(raw_idx)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise ValueError(
                    f"loss assistant turn index {raw_idx!r} out of range for {n} turns"
                )
            if idx not in indices:
                indices.append(idx)
                out.append(assistant_spans[idx])
        if not out:
            raise ValueError("loss_assistant_turns selected no assistant spans")
        return out, indices

    raise ValueError(f"unsupported loss_assistant_turns={loss_spec!r}")


def _message_text_content(msg: Dict) -> str:
    """Return the textual content rendered inside one chat message."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _assistant_texts_from_messages(messages: List[Dict]) -> List[str]:
    return [
        _message_text_content(m)
        for m in messages
        if m.get("role") == "assistant"
    ]


def _find_json_string_value_span(text: str, keys: Tuple[str, ...] = ("text", "summary")) -> Optional[Tuple[int, int]]:
    """Find the char span of a JSON string value in a tool-call payload.

    The returned span excludes the surrounding quotes, so JSON syntax tokens
    such as ``"text": "``, the closing quote, braces, and tool-call tags stay
    in the structural/closing regions.
    """
    for key in keys:
        marker = f'"{key}"'
        search_from = 0
        while True:
            key_pos = text.find(marker, search_from)
            if key_pos < 0:
                break
            pos = key_pos + len(marker)
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text) or text[pos] != ":":
                search_from = key_pos + 1
                continue
            pos += 1
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text) or text[pos] != '"':
                search_from = key_pos + 1
                continue

            body_start = pos + 1
            pos = body_start
            escaped = False
            while pos < len(text):
                ch = text[pos]
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    return body_start, pos
                pos += 1
            return None
    return None


def _token_ids_no_special(tokenizer, text: str) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _char_span_to_token_span(tokenizer, text: str, span: Tuple[int, int]) -> Optional[Tuple[int, int, int]]:
    """Map a character span in assistant text to token indices.

    Prefer tokenizer offsets when available. Fall back to prefix/body token
    lengths; that fallback is only used after the full assistant-text token
    sequence is verified against the chat-template span.
    """
    start, end = span
    try:
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc.get("offset_mapping")
        ids = enc.get("input_ids")
        if offsets is not None and ids is not None:
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            token_start = None
            token_end = None
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_end <= tok_start:
                    continue
                if tok_end > start and tok_start < end:
                    if token_start is None:
                        token_start = i
                    token_end = i + 1
            if token_start is not None and token_end is not None:
                return token_start, token_end, len(ids)
    except Exception:
        pass

    prefix_len = len(_token_ids_no_special(tokenizer, text[:start]))
    body_len = len(_token_ids_no_special(tokenizer, text[start:end]))
    total_len = len(_token_ids_no_special(tokenizer, text))
    if body_len <= 0:
        return None
    return prefix_len, min(prefix_len + body_len, total_len), total_len


def _apply_compress_token_loss_weights(
    *,
    token_loss_weight: torch.Tensor,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    messages: List[Dict],
    loss_spans: List[tuple],
    loss_turn_indices: List[int],
    tokenizer,
    data_args,
) -> Dict[str, Any]:
    """Redistribute compress loss toward action/schema/close tokens."""
    structure_w = float(getattr(data_args, "compress_structure_token_weight", 2.0) or 1.0)
    body_w = float(getattr(data_args, "compress_body_token_weight", 0.35) or 1.0)
    close_w = float(getattr(data_args, "compress_close_token_weight", 4.0) or 1.0)
    close_tail = int(getattr(data_args, "compress_close_tail_tokens", 24) or 0)

    assistant_texts = _assistant_texts_from_messages(messages)
    diagnostics = {
        "applied": False,
        "aligned": False,
        "body_tokens": 0,
        "structure_weight": structure_w,
        "body_weight": body_w,
        "close_weight": close_w,
    }

    input_ids_flat = input_ids[0].tolist()
    for (ans_start, ans_end), turn_idx in zip(loss_spans, loss_turn_indices):
        if turn_idx >= len(assistant_texts):
            continue
        assistant_text = assistant_texts[turn_idx]
        if "compress" not in assistant_text:
            continue

        # Default fallback: down-weight most of the assistant span as summary
        # body, then explicitly emphasize the opening/action prefix and the
        # closing tail. Exact alignment below will replace this with a cleaner
        # schema/body/close split.
        span_len = max(0, ans_end - ans_start)
        if span_len <= 0:
            continue
        token_loss_weight[0, ans_start: ans_end + 1] = body_w
        head_len = min(64, span_len)
        token_loss_weight[0, ans_start: ans_start + head_len] = structure_w
        if close_tail > 0:
            tail_start = max(ans_start, ans_end + 1 - close_tail)
            token_loss_weight[0, tail_start: ans_end + 1] = close_w
        else:
            token_loss_weight[0, ans_end] = close_w

        body_span = _find_json_string_value_span(assistant_text)
        assistant_ids = _token_ids_no_special(tokenizer, assistant_text)
        span_ids = input_ids_flat[ans_start:ans_end]
        exact_alignment = bool(assistant_ids) and assistant_ids == span_ids
        if body_span is not None and exact_alignment:
            mapped = _char_span_to_token_span(tokenizer, assistant_text, body_span)
            if mapped is not None:
                body_token_start, body_token_end, total_tokens = mapped
                if total_tokens == len(span_ids):
                    body_abs_start = ans_start + max(0, body_token_start)
                    body_abs_end = min(ans_start + body_token_end, ans_end)
                    if body_abs_start < body_abs_end:
                        token_loss_weight[0, ans_start: ans_end + 1] = structure_w
                        token_loss_weight[0, body_abs_start:body_abs_end] = body_w
                        # BPE often merges the final summary punctuation with
                        # the closing quote (e.g. ``."``), so include one
                        # overlapping tail token in the close region.
                        close_start = max(body_abs_end - 1, ans_start)
                        token_loss_weight[0, close_start: ans_end + 1] = close_w
                        diagnostics["aligned"] = True
                        diagnostics["body_tokens"] += int(
                            max(0, close_start - body_abs_start)
                        )

        # Never allow a labeled compress token to become zero-weight. The
        # trainer normalizes by token-weight sum, so these are relative weights
        # inside the sample rather than another sample-level multiplier.
        valid = labels[0, ans_start: ans_end + 1].ne(IGNORE_INDEX)
        local = token_loss_weight[0, ans_start: ans_end + 1]
        local[valid] = local[valid].clamp_min(0.05)
        diagnostics["applied"] = True

    return diagnostics


def preprocess_per_timestep(sample: Dict, processor, data_args=None) -> Dict:
    """Tokenize a single SFT sample (messages format) and mask labels.

    Input contract (post-pass5): sample MUST contain a ``messages`` key
    (LLaMA-Factory ShareGPT format). Each message is {"role": str,
    "content": list-of-content-items}. Loss is computed only on the
    assistant turn(s) — exact same convention as VST / DeepEyesV2 /
    Qwen-VL official finetune (scan for <|im_start|>assistant ...
    <|im_end|>, that span gets loss).

    Passes a turn-local tool schema to apply_chat_template: streaming rows get
    recall only, compression rows get compress only, and rows explicitly marked
    as recall-response/no-tools get no tool schema.
    """
    base_path = Path(sample.get("data_path", "."))

    if "messages" not in sample:
        raise ValueError(
            f"Sample {sample.get('sample_id', '?')}: missing 'messages' key. "
            f"Run scripts/agent_data/pass5_messages.py to convert "
            f"input/output samples to ShareGPT messages format."
        )
    messages = _resolve_video_paths(sample["messages"], base_path)

    # Current pass5 messages carry explicit video_metadata in type="video"
    # blocks. Keep this fallback for raw rows that still contain video items
    # without metadata.
    video_metadata = []
    has_video_meta = True
    for msg in messages:
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                meta = item.get("video_metadata")
                frames = item.get("video")
                if isinstance(meta, dict):
                    meta = {k: v for k, v in meta.items() if k != "do_sample_frames"}
                    video_metadata.append(meta)
                elif isinstance(frames, list) and frames:
                    from thinkstream.data.agent_protocol import infer_video_metadata
                    video_metadata.append(infer_video_metadata(frames))
                else:
                    has_video_meta = False

    # Tool schema: the row carries its own tools list. The pass5 renderer
    # decides which tools each trajectory needs (recall, compress, or both)
    # and writes the list inline.
    tools = sample.get("tools")
    template_kwargs = dict(
        tokenize=True, return_dict=True, return_tensors="pt",
        do_sample_frames=False,  # frame_paths are already the exact frames to use
    )
    if tools is not None:
        template_kwargs["tools"] = tools
    if video_metadata and has_video_meta:
        template_kwargs["video_metadata"] = video_metadata

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

    # Per-chunk rows carry 1 (silent/response/compress/recall_query) or 2
    # (post_recall full prefix) assistant turns. Trajectory rows
    # (``trajectory_type`` set by pass5) carry N turns covering all chunks
    # between two compress boundaries. Rows can opt into a narrower label
    # mask through ``loss_assistant_turns`` — e.g. pass5 post_recall rows
    # use ``"last"`` so the previous recall tool_call is context, not a
    # second target under a no-tools schema.
    is_trajectory_row = bool(sample.get("trajectory_type"))
    if not is_trajectory_row and len(assistant_spans) not in {1, 2}:
        sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
        raise ValueError(
            f"Sample {sid}: expected 1 or 2 assistant turn(s), "
            f"found {len(assistant_spans)}. 1 turn for "
            f"silent/response/compress; 2 turns for recall multi-turn."
        )
    if is_trajectory_row and len(assistant_spans) == 0:
        sid = sample.get("sample_id") or sample.get("trajectory_id") or "?"
        raise ValueError(
            f"Trajectory row {sid}: no assistant turns found in messages."
        )

    loss_spec = sample.get("loss_assistant_turns")
    if loss_spec is None:
        loss_spec = sample.get("loss_assistant_indices")
    loss_spans, loss_turn_indices = _select_loss_assistant_spans(
        assistant_spans, loss_spec,
    )

    for ans_start, ans_end in loss_spans:
        # ans_end is the <|im_end|> token. Train the assistant content plus
        # its end marker, but not the following newline / next role marker.
        labels[0, ans_start: ans_end + 1] = input_ids[0, ans_start: ans_end + 1]

    loss_class = sample.get("_loss_class") or _sample_loss_class(sample)
    compress_weight_diag: Optional[Dict[str, Any]] = None
    if data_args is not None:
        raw_enabled = getattr(data_args, "compress_token_weighting", True)
        if isinstance(raw_enabled, str):
            compress_token_weighting = raw_enabled.strip().lower() not in {
                "0", "false", "no", "off",
            }
        else:
            compress_token_weighting = bool(raw_enabled)
        action_class_mode = str(
            getattr(data_args, "action_class_loss_mode", "none") or "none"
        ).strip().lower()
        if compress_token_weighting or action_class_mode == "inverse_freq":
            # Always attach a token_loss_weight tensor when any weighting is
            # enabled. Mixed batches would otherwise drop token weights if
            # only some rows carried the key.
            token_loss_weight = torch.ones_like(labels, dtype=torch.float32)
            # Per-chunk compress rows: apply compress redistribution to the
            # single (or last) assistant span. Trajectory rows have N turns
            # of mixed classes; per-span class detection isn't wired yet, so
            # we skip the compress-specific token weighting for trajectory
            # rows and let action_class_loss balance compress against other
            # classes via inverse-frequency weights instead.
            if (
                not is_trajectory_row
                and compress_token_weighting
                and loss_class == "compress"
            ):
                compress_weight_diag = _apply_compress_token_loss_weights(
                    token_loss_weight=token_loss_weight,
                    input_ids=input_ids,
                    labels=labels,
                    messages=messages,
                    loss_spans=loss_spans,
                    loss_turn_indices=loss_turn_indices,
                    tokenizer=processor.tokenizer,
                    data_args=data_args,
                )
            if action_class_mode == "inverse_freq":
                # inverse-frequency-weighted: multiply class-balanced weight on top of any
                # existing compress redistribution. Per-token, not per-sample.
                #
                # Anchors come from two sources:
                #   1. Single-token action ids (<silent>, <response>, ...)
                #   2. Tool-name BPE spans inside <tool_call> JSON body
                #      ("compress", "recall"). First span token is the anchor.
                from thinkstream.sft.losses import (
                    compute_inverse_frequency_weights,
                    resolve_action_token_ids,
                    resolve_tool_call_marker_ids,
                    resolve_tool_name_token_sequences,
                )
                action_ids = resolve_action_token_ids(processor.tokenizer)
                tool_seqs = resolve_tool_name_token_sequences(processor.tokenizer)
                tc_open_ids, tc_close_ids = resolve_tool_call_marker_ids(
                    processor.tokenizer
                )
                class_weight = compute_inverse_frequency_weights(
                    labels=labels,
                    action_token_ids=action_ids,
                    tool_name_sequences=tool_seqs,
                    ignore_index=IGNORE_INDEX,
                    floor_weight=float(getattr(data_args, "action_class_weight_floor", 0.05)),
                    ceil_weight=float(getattr(data_args, "action_class_weight_ceil", 20.0)),
                    tool_call_open_ids=tc_open_ids,
                    tool_call_close_ids=tc_close_ids,
                )
                # Compose with existing weights multiplicatively. This keeps
                # compress-internal structure/body/close emphasis intact while
                # adding cross-class balance on the action keyword positions.
                token_loss_weight = token_loss_weight * class_weight
            full_result["token_loss_weight"] = token_loss_weight

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    # v12.11 P1.2 fix (2026-05-01): expose ALL assistant spans, not just the
    # first. Multi-turn recall samples have 2 assistant turns (tool_call +
    # final answer); the first-turn-only metric was missing the final answer
    # in eval. The first span is kept for backward compat; ans_spans is the
    # canonical multi-span view (used by ALL eval / metric code in v12.11+).
    ans_start, ans_end = assistant_spans[0]
    full_result["eval_meta"] = {
        "sample_type": sample.get("sample_type", "?"),
        "action": sample.get("action", ""),
        "gold_action": (sample.get("metadata") or {}).get("gold_action", ""),
        "loss_class": loss_class,
        "ans_start": ans_start,           # legacy: first span only
        "ans_end": ans_end,
        "ans_spans": list(assistant_spans),  # v12.11: all spans (1 or 2 turns)
        "loss_ans_spans": list(loss_spans),
        "loss_assistant_turn_indices": list(loss_turn_indices),
        "loss_assistant_turns": sample.get("loss_assistant_turns", "all"),
        "n_assistant_turns": len(assistant_spans),
        "sft_subtype": sample.get("sft_subtype", ""),
        "compress_token_weighting": compress_weight_diag,
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
                f"convert via:  python -m scripts.agent_data.pass5_messages"
            )

        # Default policy (DataArguments.include_failed_verification=True):
        # pass3e only TAGS verifier failures, pass4 carries them through, and
        # the trainer keeps the full trajectory so a verifier-failed turn
        # only loses CE weight via token_loss_weight (collator). Drop only
        # when an ablation explicitly opts into strict cold-start.
        # Fallback default mirrors DataArguments.include_failed_verification
        # — keep failures unless the caller proves they want strict filtering.
        include_failed = getattr(data_args, "include_failed_verification", True)
        if not include_failed:
            before = len(all_samples)
            all_samples = [
                s for s in all_samples
                if s.get("verification", {}).get("passed", True)
            ]
            failed_dropped = before - len(all_samples)
            if failed_dropped > 0:
                rank0_print(
                    f"  Dropped {failed_dropped} verification-failed samples "
                    f"(include_failed_verification=False; set =True to keep)."
                )

        # Estimate num_tokens for every sample (used for length-based filtering
        # AND HF Trainer's group_by_length sampler). Skipping this leaves every
        # sample with default 3500 → batches are wildly heterogeneous → padding
        # waste + silent overflow.
        for s in all_samples:
            if "num_tokens" not in s:
                s["num_tokens"] = _estimate_sample_tokens(s)
            s["_has_visual"] = _sample_has_visual(s)

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
        # Deterministic subsample so train logs stay comparable across runs.
        # By default this preserves the historical Random(0) natural sample.
        # When eval_balance_strategy is set, use a class-balanced subset for
        # checkpoint selection while keeping full/natural eval available by
        # setting eval_balance_strategy=none or eval_max_samples=0.
        if max_samples is not None and max_samples > 0 and len(all_samples) > max_samples:
            all_samples = _subsample_eval_balanced(all_samples, max_samples, data_args)
            rank0_print(f"  Subsampled eval set to {len(all_samples)}")

        _assign_class_loss_weights(all_samples, data_args)

        rank0_print(f"Total samples: {len(all_samples)}")

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.data_args = data_args
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
        # Positive = visual row, negative = text-only row.
        # WeightedSFTTrainer's modality-grouped sampler uses the sign to
        # keep ZeRO3 ranks on the same module path. New compress rows carry
        # visual_window, but legacy/generated diagnostic rows may still be
        # text-only.
        return [
            s["num_tokens"] if s.get("_has_visual", True) else -s["num_tokens"]
            for s in self.samples
        ]

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

        # Tokenize + vision + label mask. apply_chat_template uses the
        # turn-local tool schema carried by the sample.
        data_dict = preprocess_per_timestep(sample, self.processor, self.data_args)

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
            "loss_class": sample.get("_loss_class") or _sample_loss_class(sample),
            "action": sample.get("action"),
            "sequence_type": sample.get("sequence_type"),
            "base_role": sample.get("base_role"),
        }
        data_dict["sample_weights"] = torch.tensor(
            float(sample.get("_sample_weight", 1.0)),
            dtype=torch.float32,
        )

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

    Emits ``video_mask`` ([B, L] bool, True where the input is a
    ``<|video_pad|>`` token) so the patched lce_forward can build the
    sliding-window FlexAttention block mask. The mask is data-side: it
    activates only when ``attn_implementation="streaming_attention"`` is
    in use (otherwise the model forward ignores ``video_mask`` entirely),
    so emitting it unconditionally is safe.
    """

    tokenizer: transformers.PreTrainedTokenizer
    emit_video_mask: bool = False
    _video_token_id: Optional[int] = None

    def __post_init__(self):
        # Resolve once at collator construction. Qwen3-VL uses ``<|video_pad|>``
        # for the per-frame placeholder token; rope2d.py hardcodes 151656 as a
        # known constant, but tokenizer lookup keeps us safe against future
        # vocab shifts. Only consumed when ``emit_video_mask=True`` —
        # streaming_attention path needs it; standard / flash_attention_2
        # paths reject unknown kwargs.
        if self.emit_video_mask:
            try:
                tid = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
                self._video_token_id = (
                    int(tid) if tid is not None and tid >= 0 else None
                )
            except Exception:
                self._video_token_id = None
            if self._video_token_id is None:
                logging.warning(
                    "PerTimestepDataCollator(emit_video_mask=True): could not "
                    "resolve <|video_pad|> token id; video_mask will not be "
                    "emitted. streaming_attention will fall back to causal."
                )

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
        if self.emit_video_mask and self._video_token_id is not None:
            # bool tensor; same shape as input_ids. Consumed by the patched
            # qwen3_vl.lce_forward → build_video_block_mask. Emitted only
            # when streaming_attention is wired so non-flex forward paths
            # (which reject unknown kwargs) stay happy.
            batch["video_mask"] = input_ids == self._video_token_id
        if token_loss_weight is not None:
            batch["token_loss_weight"] = token_loss_weight

        sample_weights = [
            inst["sample_weights"].reshape(()) for inst in instances
            if "sample_weights" in inst
        ]
        if len(sample_weights) == len(instances):
            batch["sample_weights"] = torch.stack(sample_weights).float()

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

def make_per_timestep_data_module(
    processor, data_args, *, emit_video_mask: bool = False,
) -> Dict:
    """Create dataset + collator for per-timestep agent SFT.

    Builds an eval_dataset when DataArguments.eval_dataset_use is set —
    typically `stream_agent_val` (held-out video-disjoint pool). The HF
    Trainer then runs eval on this every --eval_steps to surface
    overfitting in real time.

    ``emit_video_mask`` is set by train.py when
    ``attn_implementation="streaming_attention"`` so the collator emits a
    per-token bool tensor identifying ``<|video_pad|>`` tokens for the
    FlexAttention block-mask builder.
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

    collator = PerTimestepDataCollator(
        processor.tokenizer, emit_video_mask=emit_video_mask,
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
