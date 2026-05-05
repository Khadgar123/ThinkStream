"""Standalone pass2 prompt A/B harness for repair-heavy batch videos.

This script re-runs pass2 teacher rollouts on a small set of worst-case
videos and compares prompt / memory / ordering variants without touching the
production pipeline.

Default intent:
- pick top repair-rejected videos from `batch1/audits/pass2_chunks.jsonl`
- run the first ~40 chunks of each video
- compare memory serialization, text-vs-visual block order, and frame order
- report stale/repair metrics into an output directory

Historical context:
- 2026-05-04 one-off prompt probes on `v_9xtYwXpaiZ0` chunk 112 showed the
  old full-memory prose prompt could copy stale shoe-polishing text, while
  current-first prompts with no memory / current-only visual evidence were
  robust. This harness generalizes that test to full partial-rollouts on the
  repair-heavy batch1 videos.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.agent_data_pipeline.vllm_client import VLLMClient, encode_image_base64
from thinkstream.data.agent_protocol import append_timestamped_image_list

from .audit_pass2_stale import audit_rollouts
from .config import (
    AGENT_CHUNK_SEC,
    COMPRESS_RANGE_MIN,
    FRAMES_PER_CHUNK,
    PASS_CONFIG,
    RUNTIME_MM_PROCESSOR_KWARGS,
    SUMMARY_TOKENS_MAX,
    VLLM_MODEL,
    compute_visual_window_start,
)
from .pass1a_evidence import get_chunk_frame_paths
from .pass2_rollout import (
    MemoryState,
    _chunk_visual_delta_mse,
    _is_repair_better,
    _safe_max_tokens_for_pass2,
    build_compress_request,
    parse_compress_result,
    parse_observation_result,
    should_repair_observation,
)

logger = logging.getLogger(__name__)

PROSE_OBSERVATION_PROMPT = """You are a streaming video agent generating a think note.

CURRENT TASK FIRST: inspect the timestamp-tagged image list for the sliding visual window t={window_start}-{window_end}s. The latest target chunk is ONLY t={start}-{end}s ({current_frame_count} frames) and is the primary evidence.

Memory below is untrusted history for entity naming only. It may describe older frames and must not be copied if the latest frames differ.
<memory>
{recent_thinks}
</memory>

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

The timestamp-tagged images are ordered from older context to the latest chunk.

Evidence priority:
1. The tagged images at t={start}-{end}s are the only evidence for the current think.
2. Older tagged images are context only.
3. Memory is history and entity naming only. Ignore memory when it conflicts with the latest frames.
4. Never use memory as evidence that a past object/action is still visible.

Rules:
- Ground the note only in observable visual facts from the latest target chunk
- Do not copy any XML-like tag, timestamp marker, role marker, or metadata line into the output
- Mention current OCR, logos, icons, labels, title cards, graphic overlays, and spatial layout when visible
- Reuse a memory entity phrase only when that same entity is visibly present now
- Do not copy a prior sentence or mention any object/action from memory unless it is visible in the latest target chunk
- If memory says a person/hand is holding, pressing, pouring, cutting, walking, or otherwise manipulating something, write that action only when the actor and contact/motion are visible in the latest target chunk
- If the latest frames show an object at rest, on a stand/table/surface, or as a static screen/card, describe that current state directly instead of repeating an old manipulation
- If the latest frames show a different object/action, title card, branding card, transition card, or static graphic, name it directly
- Avoid "continues", "remains", "persists", "still", "same", and "without change" unless those words are justified by the latest target chunk alone
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

PROSE_REPAIR_PROMPT = """You are correcting a streaming video think note for one current chunk.

Recent memory/entity names (may be stale; use only for naming):
{recent_thinks}

Previous stale draft to avoid copying:
{stale_text}

The tagged frames below contain ONLY the current 1 second: t={start}-{end}s
({n_frames} frames at {fps} fps).

Task: inspect the current tagged frames first and write the actual visual note for
t={start}-{end}s.

Evidence priority:
1. Current tagged frames at t={start}-{end}s.
2. Memory/entity names only if the same entity is visibly present.
3. Never use memory or the stale draft as evidence for what is visible now.

Rules:
- Describe only observable visual facts in this 1-second chunk
- Do not copy any XML-like tag, timestamp marker, role marker, or metadata line into the output
- Keep entity names consistent only when the same entity is visibly present
- Do not say "continues", "remains", "unchanged", or "no new" unless the
  current frames visibly show the same object/action
- If the current frames show a new object/action, name that directly
- NO meta-reasoning, NO sounds/smells/emotions
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

CURRENT_ONLY_PROMPT = """You are a streaming video agent generating a think note.

The timestamp-tagged image list contains ONLY the current 1 second t={start}-{end}s ({current_frame_count} frames).

Describe the concrete visual contents of these frames. Include visible OCR, logos, icons, labels, title cards, and graphic overlays if present.

Rules:
- Use only what is visible in these frames
- Do not use any history or prior draft
- Do not copy any XML-like tag, timestamp marker, role marker, or metadata line into the output
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

CURRENT_ONLY_REPAIR_PROMPT = """You are correcting a streaming video think note for one current chunk.

Ignore all history and ignore the stale draft.

Previous stale draft to avoid copying:
{stale_text}

The tagged frames below contain ONLY the current 1 second: t={start}-{end}s
({n_frames} frames at {fps} fps).

Rules:
- Describe only observable visual facts in this 1-second chunk
- Mention visible OCR, logos, icons, labels, title cards, and graphic overlays
- Do not copy any XML-like tag, timestamp marker, role marker, or metadata line into the output
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

VISUAL_ONLY_2FRAME_REPAIR_PROMPT = """You are generating a corrected think note for one current chunk.

The tagged frames below contain ONLY the current 1 second: t={start}-{end}s
({n_frames} frames at {fps} fps).

Rules:
- Use only what is visible in these current frames
- Do not use any history or prior draft
- Mention current OCR, logos, icons, labels, title cards, and graphic overlays when visible
- Do not copy any XML-like tag, timestamp marker, role marker, or metadata line into the output
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

STUDENT_TAGS_OBSERVATION_PROMPT = """You are a streaming video agent generating a think note for one current 1-second chunk.

CURRENT TASK FIRST: inspect the timestamp-tagged image list for the sliding visual window t={window_start}-{window_end}s. The latest target chunk is ONLY t={start}-{end}s ({current_frame_count} frames) and is the primary evidence.

Past text memory is shown below. Each <memory t="...">...</memory> line is older text from earlier times and may be stale.
{recent_thinks}

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

The timestamp-tagged images are ordered from older context to the latest chunk. Frames labeled t={start}-{end}s are the only evidence for the current think; older timestamps are context only.

Evidence priority:
1. The tagged images at t={start}-{end}s are the only evidence for the current think.
2. Older tagged images are context only.
3. Past text memory is for naming only. Ignore it when it conflicts with the latest frames.
4. Never use past text memory as evidence that a past object/action is still visible now.

Rules:
- Ground the note only in observable visual facts from the latest target chunk
- Do not copy any XML-like tag, timestamp marker, role marker, metadata line, or memory text into the output
- Mention current OCR, logos, icons, labels, title cards, graphic overlays, and spatial layout when visible
- Reuse a memory phrase only when that same entity is visibly present now
- Do not copy a prior sentence or mention any object/action from memory unless it is visible in the latest target chunk
- If memory says a person/hand is holding, pressing, pouring, cutting, walking, or otherwise manipulating something, write that action only when the actor and contact/motion are visible in the latest target chunk
- If the latest frames show an object at rest, on a stand/table/surface, or as a static screen/card, describe that current state directly instead of repeating an old manipulation
- If the latest frames show a different object/action, title card, branding card, transition card, or static graphic, name it directly
- Avoid "continues", "remains", "persists", "still", "same", and "without change" unless those words are justified by the latest target chunk alone
- Final self-check before answering: if a phrase came from the memory lines rather than the latest two frames, rewrite it
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

STUDENT_TAGS_OBSERVATION_PROMPT_MAX_STRICT = """You are a streaming video agent generating a think note for one current 1-second chunk.

CURRENT TASK FIRST: inspect the timestamp-tagged image list for the sliding visual window t={window_start}-{window_end}s. The latest target chunk is ONLY t={start}-{end}s ({current_frame_count} frames) and is the primary evidence.

Past text memory is shown below. Each <memory t="...">...</memory> line is older text from earlier times and may be stale.
{recent_thinks}

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

The timestamp-tagged images are ordered from older context to the latest chunk. Frames labeled t={start}-{end}s are the only evidence for the current think; older timestamps are context only.

Required workflow:
1. First isolate the two frames tagged t={start}-{end}s and draft the note from those frames alone.
2. Only after drafting, look at older window frames for coarse scene continuity.
3. Use past text memory only to recover a stable noun phrase for the same visibly present entity.

Evidence priority:
1. The tagged images at t={start}-{end}s are the only evidence for the current think.
2. Older tagged images are coarse context only and cannot supply the current action.
3. Past text memory is never continuation context. It is naming-only and cannot supply any current action, state, color, count, position, relation, or event.

Rules:
- Ground the note only in observable visual facts from the latest target chunk
- Do not copy any XML-like tag, timestamp marker, role marker, metadata line, or memory text into the output
- Mention current OCR, logos, icons, labels, title cards, graphic overlays, and spatial layout when visible
- You may reuse only a stable noun phrase from memory when that same entity is visibly present now
- Never reuse or paraphrase a past sentence, action phrase, adjective phrase, spatial relation, or event description from memory
- Even when the same object is still visible now, rewrite from the current two frames in fresh wording
- If the current two frames do not clearly show contact or motion, describe the visible present state instead of inheriting a past action
- Default to concrete present-state description, not continuity wording
- Do not use "continues", "remains", "still", "same", "ongoing", "no significant changes", or similar continuity words unless both current frames themselves directly support that wording
- If a phrase could have been written from memory without looking at the current two frames, delete or rewrite it
- NO meta-reasoning, NO "I notice", NO sounds/smells/emotions
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

STUDENT_TAGS_OBSERVATION_PROMPT_MILD = """You are a streaming video agent generating a think note for one current 1-second chunk.

CURRENT TASK FIRST: inspect the timestamp-tagged image list for the sliding visual window t={window_start}-{window_end}s. The latest target chunk is ONLY t={start}-{end}s ({current_frame_count} frames).

Past text memory is shown below. Each <memory t="...">...</memory> line is older text from earlier times.
{recent_thinks}

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text.

The timestamp-tagged images are ordered from older context to the latest chunk. Use the latest target chunk as the main evidence. Older window frames are context. Past text memory may help keep names consistent when the same entity is still visible.

Rules:
- Describe observable visual facts from the latest target chunk
- Mention current OCR, logos, icons, labels, title cards, and overlays when visible
- Do not copy XML-like tags or timestamp markers into the output
- If the latest frames show a different object/action, describe what is visible now
- One paragraph, 40-80 tokens, never exceed 100

Output one paragraph:"""

STUDENT_TAGS_REPAIR_PROMPT = """You are correcting a streaming video think note for one current chunk.

Past text memory lines from earlier times (may be stale; use only for naming if the same entity is visibly present):
{recent_thinks}

Previous stale draft to avoid copying:
{stale_text}

The tagged frames below contain ONLY the current 1 second: t={start}-{end}s
({n_frames} frames at {fps} fps).

Each image is preceded by a structural tag like <frame ts="12.5" role="latest chunk" />. These frame tags are routing metadata only, not answer text. Never copy or paraphrase any frame tag in the output.

Task: inspect the current tagged frames first and write the actual visual note for
t={start}-{end}s.

Evidence priority:
1. Current tagged frames at t={start}-{end}s.
2. Memory/entity names only if the same entity is visibly present now.
3. Never use memory or the stale draft as evidence for what is visible now.

Rules:
- Describe only observable visual facts in this 1-second chunk
- Do not copy any XML-like tag, timestamp marker, role marker, metadata line, or memory text into the output
- Keep entity names consistent only when the same entity is visibly present
- Do not say "continues", "remains", "unchanged", or "no new" unless the current frames visibly show the same object/action
- If the current frames show a new object/action, name that directly
- 40-80 tokens, one paragraph, no meta-reasoning

Output one paragraph:"""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    memory_mode: str
    block_order: str
    frame_order: str
    visual_scope: str = "window"
    prompt_mode: str = "structured"
    memory_recent_limit: Optional[int] = None
    memory_include_summaries: bool = True
    memory_order: str = "timeline"
    instruction_mode: str = "strong"
    repair_mode: str = "aligned"
    prepend_latest_chunk_first: bool = False


DEFAULT_VARIANTS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        name="structured_text_first_forward",
        memory_mode="structured",
        block_order="text_first",
        frame_order="forward",
    ),
    VariantSpec(
        name="structured_text_first_reverse",
        memory_mode="structured",
        block_order="text_first",
        frame_order="reverse",
    ),
    VariantSpec(
        name="structured_images_first_forward",
        memory_mode="structured",
        block_order="images_first",
        frame_order="forward",
    ),
    VariantSpec(
        name="structured_minfields_text_first_forward",
        memory_mode="structured_minfields",
        block_order="text_first",
        frame_order="forward",
    ),
    VariantSpec(
        name="prose_text_first_forward",
        memory_mode="prose",
        block_order="text_first",
        frame_order="forward",
        prompt_mode="prose",
    ),
    VariantSpec(
        name="no_history_text_first_forward",
        memory_mode="none",
        block_order="text_first",
        frame_order="forward",
    ),
    VariantSpec(
        name="student_tags_last2_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=2,
        memory_include_summaries=True,
        memory_order="timeline",
    ),
    VariantSpec(
        name="student_tags_full_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="timeline",
    ),
    VariantSpec(
        name="student_tags_summary_only_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=0,
        memory_include_summaries=True,
        memory_order="timeline",
    ),
    VariantSpec(
        name="fulltag_mild_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="timeline",
        instruction_mode="mild",
        repair_mode="current_only_visual",
    ),
    VariantSpec(
        name="fulltag_strong_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="timeline",
        instruction_mode="strong",
        repair_mode="current_only_visual",
    ),
    VariantSpec(
        name="fulltag_max_strict_timeline_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="timeline",
        instruction_mode="max_strict",
        repair_mode="current_only_visual",
    ),
    VariantSpec(
        name="fulltag_strong_summaries_first_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="summaries_first",
        instruction_mode="strong",
        repair_mode="current_only_visual",
    ),
    VariantSpec(
        name="fulltag_strong_timeline_dup_latest_text_first_forward",
        memory_mode="student_tags",
        block_order="text_first",
        frame_order="forward",
        memory_recent_limit=None,
        memory_include_summaries=True,
        memory_order="timeline",
        instruction_mode="strong",
        repair_mode="current_only_visual",
        prepend_latest_chunk_first=True,
    ),
)


def _iter_recent_thinks(memory: MemoryState, limit: Optional[int] = None) -> List[Dict]:
    thinks = list(memory.recent_thinks)
    if limit is None:
        return thinks
    return thinks[-max(0, int(limit)) :]


def _select_memory_items(
    memory: MemoryState,
    *,
    include_summaries: bool,
    recent_limit: Optional[int],
    memory_order: str,
) -> List[Dict]:
    if memory_order not in {"timeline", "summaries_first"}:
        raise ValueError(f"Unsupported memory_order={memory_order!r}")

    allowed_recent: Optional[set[int]] = None
    if recent_limit is not None:
        if int(recent_limit) <= 0:
            allowed_recent = set()
        else:
            allowed_recent = {
                int(item.get("chunk", -1))
                for item in _iter_recent_thinks(memory, limit=recent_limit)
            }

    def _keep(item: Dict) -> bool:
        if item.get("type") == "summary":
            return include_summaries
        if allowed_recent is None:
            return True
        return int(item.get("chunk", -1)) in allowed_recent

    if memory_order == "timeline":
        return [item for item in memory.timeline if _keep(item)]

    items: List[Dict] = []
    if include_summaries:
        items.extend(memory.compressed_segments)
    if allowed_recent is None:
        items.extend(memory.recent_thinks)
    elif allowed_recent:
        items.extend(
            item for item in memory.recent_thinks
            if int(item.get("chunk", -1)) in allowed_recent
        )
    return items


def _serialize_student_tags_memory(
    memory: MemoryState,
    *,
    include_summaries: bool,
    recent_limit: Optional[int],
    memory_order: str,
) -> str:
    lines: List[str] = []
    for item in _select_memory_items(
        memory,
        include_summaries=include_summaries,
        recent_limit=recent_limit,
        memory_order=memory_order,
    ):
        lines.append(MemoryState._format_timeline_item_as_memory_tag(item))
    return "\n".join(lines) or "(none)"


def _serialize_memory(
    memory: MemoryState,
    mode: str,
    *,
    recent_limit: Optional[int] = None,
    include_summaries: bool = True,
    memory_order: str = "timeline",
) -> str:
    if mode == "structured":
        return memory.format_for_observation_prompt() or "(none)"
    if mode == "structured_minfields":
        records: List[str] = []
        for item in memory.timeline:
            record = {"kind": item.get("type")}
            if item.get("type") == "summary":
                record["time_range"] = list(item.get("time_range") or [])
            else:
                record["chunk"] = int(item.get("chunk", -1))
                record["time"] = item.get("time", "")
            record["text"] = item.get("text", "")
            records.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(records) or "(none)"
    if mode == "student_tags":
        return _serialize_student_tags_memory(
            memory,
            include_summaries=include_summaries,
            recent_limit=recent_limit,
            memory_order=memory_order,
        )
    if mode == "prose":
        return memory.format_for_prompt() or "(none)"
    if mode == "recent2":
        temp = MemoryState()
        temp.timeline = _iter_recent_thinks(memory, limit=2)
        return temp.format_for_prompt() or "(none)"
    if mode == "none":
        return "(none)"
    raise ValueError(f"Unsupported memory_mode={mode!r}")


def _serialize_recent_history(
    memory: MemoryState,
    mode: str,
    limit: int = 8,
    *,
    include_summaries: bool = False,
    memory_order: str = "timeline",
) -> str:
    if mode == "structured":
        return memory.format_recent_for_repair_prompt(limit=limit)
    if mode == "structured_minfields":
        records: List[str] = []
        for item in _iter_recent_thinks(memory, limit=limit):
            record = {
                "kind": "think",
                "chunk": int(item.get("chunk", -1)),
                "time": item.get("time", ""),
                "text": item.get("text", ""),
            }
            records.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(records) or "(none)"
    if mode == "student_tags":
        return _serialize_student_tags_memory(
            memory,
            include_summaries=include_summaries,
            recent_limit=limit,
            memory_order=memory_order,
        )
    if mode in {"prose", "recent2"}:
        lines = [
            f'[{item["time"]}] {item.get("text", "")}'
            for item in _iter_recent_thinks(memory, limit=limit)
        ]
        return "\n".join(lines) or "(none)"
    if mode == "none":
        return "(none)"
    raise ValueError(f"Unsupported memory_mode={mode!r}")


def _variant_prompt(spec: VariantSpec) -> str:
    if spec.memory_mode == "student_tags":
        if spec.instruction_mode == "mild":
            return STUDENT_TAGS_OBSERVATION_PROMPT_MILD
        if spec.instruction_mode == "max_strict":
            return STUDENT_TAGS_OBSERVATION_PROMPT_MAX_STRICT
        return STUDENT_TAGS_OBSERVATION_PROMPT
    if spec.prompt_mode == "structured":
        if spec.visual_scope == "current_only":
            return CURRENT_ONLY_PROMPT
        from .config import OBSERVATION_PROMPT

        return OBSERVATION_PROMPT
    if spec.prompt_mode == "prose":
        return CURRENT_ONLY_PROMPT if spec.visual_scope == "current_only" else PROSE_OBSERVATION_PROMPT
    raise ValueError(f"Unsupported prompt_mode={spec.prompt_mode!r}")


def _variant_repair_prompt(spec: VariantSpec) -> str:
    if spec.repair_mode == "current_only_visual":
        return VISUAL_ONLY_2FRAME_REPAIR_PROMPT
    if spec.memory_mode == "student_tags":
        return STUDENT_TAGS_REPAIR_PROMPT
    if spec.prompt_mode == "structured":
        if spec.visual_scope == "current_only":
            return CURRENT_ONLY_REPAIR_PROMPT
        from .config import OBSERVATION_REPAIR_PROMPT

        return OBSERVATION_REPAIR_PROMPT
    if spec.prompt_mode == "prose":
        return CURRENT_ONLY_REPAIR_PROMPT if spec.visual_scope == "current_only" else PROSE_REPAIR_PROMPT
    raise ValueError(f"Unsupported prompt_mode={spec.prompt_mode!r}")


def _ordered_frames(
    frame_paths: List[str],
    chunk_idx: int,
    *,
    visual_scope: str,
    frame_order: str,
) -> Tuple[List[str], List[str], int]:
    if visual_scope == "current_only":
        window_start = chunk_idx
    elif visual_scope == "window":
        window_start = compute_visual_window_start(chunk_idx)
    else:
        raise ValueError(f"Unsupported visual_scope={visual_scope!r}")

    window_images: List[str] = []
    timestamp_labels: List[str] = []
    for c in range(window_start, chunk_idx + 1):
        label = "latest chunk" if c == chunk_idx else "older context"
        for img_path in get_chunk_frame_paths(frame_paths, c):
            if not Path(img_path).exists():
                continue
            window_images.append(img_path)
            timestamp_labels.append(label)

    if frame_order == "reverse":
        window_images.reverse()
        timestamp_labels.reverse()
    elif frame_order != "forward":
        raise ValueError(f"Unsupported frame_order={frame_order!r}")

    return window_images, timestamp_labels, window_start


def build_variant_observation_request(
    spec: VariantSpec,
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
) -> Dict:
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    window_images, timestamp_labels, window_start = _ordered_frames(
        frame_paths,
        chunk_idx,
        visual_scope=spec.visual_scope,
        frame_order=spec.frame_order,
    )
    prompt = _variant_prompt(spec).format(
        compressed_memory="(see memory timeline below)",
        recent_thinks=_serialize_memory(
            memory,
            spec.memory_mode,
            recent_limit=spec.memory_recent_limit,
            include_summaries=spec.memory_include_summaries,
            memory_order=spec.memory_order,
        ),
        window_start=int(window_start * AGENT_CHUNK_SEC),
        window_end=int(end),
        start=int(start),
        end=int(end),
        current_frame_count=FRAMES_PER_CHUNK,
    )

    prompt_item = {"type": "text", "text": prompt}
    frame_content: List[Dict] = []
    if spec.prepend_latest_chunk_first:
        latest_chunk_paths = [
            p for p in get_chunk_frame_paths(frame_paths, chunk_idx) if Path(p).exists()
        ]
        append_timestamped_image_list(
            frame_content,
            latest_chunk_paths,
            fps=float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC),
            start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
            total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
            context_label="latest chunk",
            image_key="image_url",
            image_url_encoder=encode_image_base64,
        )
    append_timestamped_image_list(
        frame_content,
        window_images,
        fps=float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC),
        start_frame_index=window_start * FRAMES_PER_CHUNK,
        total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
        timestamp_labels=timestamp_labels,
        image_key="image_url",
        image_url_encoder=encode_image_base64,
    )
    content = [prompt_item] + frame_content
    if spec.block_order == "images_first":
        content = frame_content + [prompt_item]
    elif spec.block_order != "text_first":
        raise ValueError(f"Unsupported block_order={spec.block_order!r}")

    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_observation"],
        "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
        "id": f"{video_id}_{spec.name}_obs_{chunk_idx}",
    }


def build_variant_repair_request(
    spec: VariantSpec,
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
    *,
    stale_text: str,
) -> Dict:
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    chunk_frame_paths = [
        p for p in get_chunk_frame_paths(frame_paths, chunk_idx) if Path(p).exists()
    ]
    if spec.frame_order == "reverse":
        chunk_frame_paths = list(reversed(chunk_frame_paths))
    fps = float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)
    prompt = _variant_repair_prompt(spec).format(
        recent_thinks=_serialize_recent_history(
            memory,
            spec.memory_mode,
            limit=8,
            include_summaries=(spec.repair_mode != "current_only_visual"),
            memory_order=spec.memory_order,
        ),
        stale_text=(stale_text or "").strip()[:600] or "(none)",
        start=int(start),
        end=int(end),
        n_frames=len(chunk_frame_paths),
        fps=fps,
        current_frame_count=len(chunk_frame_paths),
    )
    prompt_item = {"type": "text", "text": prompt}
    frame_content: List[Dict] = []
    append_timestamped_image_list(
        frame_content,
        chunk_frame_paths,
        fps=fps,
        start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
        total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
        context_label="latest chunk",
        image_key="image_url",
        image_url_encoder=encode_image_base64,
    )
    content = [prompt_item] + frame_content
    if spec.block_order == "images_first":
        content = frame_content + [prompt_item]
    elif spec.block_order != "text_first":
        raise ValueError(f"Unsupported block_order={spec.block_order!r}")

    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_observation"],
        "temperature": min(float(PASS_CONFIG["pass2_rollout"]["temperature"]), 0.2),
        "id": f"{video_id}_{spec.name}_obs_repair_{chunk_idx}",
    }


def select_repair_heavy_videos(
    chunk_log_path: Path,
    *,
    top_k: int,
) -> List[str]:
    rejected: Dict[str, int] = {}
    attempted: Dict[str, int] = {}
    chunks: Dict[str, int] = {}
    with chunk_log_path.open() as f:
        for line in f:
            obj = json.loads(line)
            vid = str(obj["video_id"])
            chunks[vid] = chunks.get(vid, 0) + 1
            if obj.get("repair_attempted"):
                attempted[vid] = attempted.get(vid, 0) + 1
            if obj.get("repair_rejected"):
                rejected[vid] = rejected.get(vid, 0) + 1

    ordered = sorted(
        rejected,
        key=lambda vid: (rejected.get(vid, 0), attempted.get(vid, 0), chunks.get(vid, 0)),
        reverse=True,
    )
    return ordered[:top_k]


def _load_evidence(batch_root: Path, video_id: str) -> Optional[List[Dict]]:
    for subdir in ("evidence_1b", "evidence_1a"):
        path = batch_root / subdir / f"{video_id}.json"
        if path.exists():
            return json.loads(path.read_text())
    return None


def _frame_paths_for_video(batch_root: Path, video_id: str) -> List[str]:
    frame_dir = batch_root / "frames" / video_id
    return [str(p) for p in sorted(frame_dir.glob("frame_*.jpg"))]


def _max_repeat_run(texts: Sequence[str], *, near: bool) -> int:
    if not texts:
        return 0

    def _norm(text: str) -> str:
        import re

        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _tokens(text: str) -> set[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "into", "over",
            "under", "left", "right", "center", "middle", "frame", "video", "scene",
            "visible", "text", "white", "black", "still", "same", "latest", "second",
            "continues", "continue", "remain", "remains", "unchanged", "static",
            "during", "throughout", "current",
        }
        return {
            w for w in _norm(text).split() if len(w) > 2 and w not in stop
        }

    def _jaccard(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(len(a | b), 1)

    best = 1
    if near:
        tok = [_tokens(t) for t in texts]
        start = 0
        for i in range(1, len(texts) + 1):
            keep = i < len(texts) and _jaccard(tok[i - 1], tok[i]) >= 0.86
            if not keep:
                best = max(best, i - start)
                start = i
        return best

    norms = [_norm(t) for t in texts]
    start = 0
    for i in range(1, len(norms) + 1):
        if i == len(norms) or norms[i] != norms[start]:
            best = max(best, i - start)
            start = i
    return best


async def run_variant_single_video(
    *,
    spec: VariantSpec,
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client: VLLMClient,
    evidence: Optional[List[Dict]],
    chunk_log_path: Path,
) -> Dict:
    memory = MemoryState()
    thinks: List[Dict] = []
    compression_events: List[Dict] = []
    snapshots: Dict[int, Dict] = {}

    for chunk_idx in range(num_chunks):
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)
        pre_action_timeline = snapshots[chunk_idx]["timeline"]
        pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]
        should_compress_now = (
            memory.should_compress() and len(pre_action_thinks) >= COMPRESS_RANGE_MIN
        )

        request = build_variant_observation_request(
            spec, chunk_idx, frame_paths, memory, video_id
        )
        safe_obs_max = _safe_max_tokens_for_pass2(request, request["max_tokens"])
        mm_kwargs = dict(RUNTIME_MM_PROCESSOR_KWARGS)
        mm_kwargs["do_sample_frames"] = False
        raw = await client._call_one(
            messages=request["messages"],
            max_tokens=safe_obs_max,
            temperature=request["temperature"],
            request_id=request["id"],
            enable_thinking=bool(PASS_CONFIG["pass2_rollout"].get("thinking", False)),
            mm_processor_kwargs=mm_kwargs,
        )
        think_text = parse_observation_result(raw)

        repaired = False
        repair_attempted = False
        repair_rejected = False
        repair_meta: Dict = {}
        visual_delta_mse = _chunk_visual_delta_mse(frame_paths, chunk_idx)
        should_repair, repair_meta = should_repair_observation(
            think_text,
            memory.recent_thinks,
            chunk_idx=chunk_idx,
            evidence=evidence,
            visual_delta_mse=visual_delta_mse,
        )
        if should_repair:
            repair_attempted = True
            repair_request = build_variant_repair_request(
                spec,
                chunk_idx,
                frame_paths,
                memory,
                video_id,
                stale_text=think_text,
            )
            safe_repair_max = _safe_max_tokens_for_pass2(
                repair_request, repair_request["max_tokens"]
            )
            repair_raw = await client._call_one(
                messages=repair_request["messages"],
                max_tokens=safe_repair_max,
                temperature=repair_request["temperature"],
                request_id=repair_request["id"],
                enable_thinking=bool(PASS_CONFIG["pass2_rollout"].get("thinking", False)),
                mm_processor_kwargs=mm_kwargs,
            )
            repaired_text = parse_observation_result(repair_raw)
            if _is_repair_better(repaired_text, think_text, memory.recent_thinks):
                think_text = repaired_text
                repaired = True
            else:
                repair_rejected = True

        think_record = {
            "chunk_idx": chunk_idx,
            "time": [chunk_idx * AGENT_CHUNK_SEC, (chunk_idx + 1) * AGENT_CHUNK_SEC],
            "think": think_text,
        }
        if repaired:
            think_record["repair"] = repair_meta
        thinks.append(think_record)

        if should_compress_now:
            comp_request = build_compress_request(
                pre_action_timeline,
                memory,
                video_id,
                chunk_idx,
                evidence=evidence,
                frame_paths=frame_paths,
            )
            if comp_request is None:
                memory.add_think(chunk_idx, think_text)
            else:
                safe_comp_max = _safe_max_tokens_for_pass2(
                    comp_request, comp_request["max_tokens"]
                )
                comp_raw = await client._call_one(
                    messages=comp_request["messages"],
                    max_tokens=safe_comp_max,
                    temperature=comp_request["temperature"],
                    request_id=f"{video_id}_{spec.name}_compress_{chunk_idx}",
                    enable_thinking=bool(PASS_CONFIG["pass2_rollout"].get("thinking", False)),
                )
                summary = parse_compress_result(comp_raw, comp_request["_meta"])
                selected_indices = comp_request["_meta"]["selected_indices"]
                memory.compress(summary, selected_indices=selected_indices)
                memory.add_think(chunk_idx, think_text)
                compression_events.append(
                    {
                        "trigger_chunk": chunk_idx,
                        "summary": summary,
                        "selected_indices": selected_indices,
                        "compressed_thinks_chunks": comp_request["_meta"].get("chunks", []),
                        "teacher_policy": comp_request["_meta"].get("teacher_policy", {}),
                        "post_compress_tokens": memory.count_recent_tokens(),
                    }
                )
        else:
            memory.add_think(chunk_idx, think_text)

        chunk_entry = {
            "variant": spec.name,
            "video_id": video_id,
            "chunk": chunk_idx,
            "think": think_text[:160],
            "tokens": memory.count_tokens(),
            "compressed": should_compress_now,
            "repair_attempted": repair_attempted,
            "repair_rejected": repair_rejected,
            "repaired": repaired,
        }
        if repair_meta:
            chunk_entry["repair_reason"] = repair_meta.get("reason")
            chunk_entry["repair_run_length"] = repair_meta.get("run_length")
            chunk_entry["repair_evidence_drift"] = repair_meta.get("evidence_drift")
        if visual_delta_mse is not None:
            chunk_entry["visual_delta_mse"] = round(visual_delta_mse, 3)
        with chunk_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(chunk_entry, ensure_ascii=False) + "\n")

    return {
        "video_id": video_id,
        "variant": spec.name,
        "num_chunks": num_chunks,
        "thinks": thinks,
        "compression_events": compression_events,
        "snapshots": snapshots,
        "final_memory": memory.snapshot(num_chunks),
    }


def summarize_variant_video(
    *,
    spec: VariantSpec,
    rollout: Dict,
    evidence: Optional[List[Dict]],
    min_run_chunks: int,
    drift_threshold: float,
) -> Dict:
    video_id = str(rollout["video_id"])
    texts = [str(t.get("think", "")) for t in rollout.get("thinks") or []]
    audit = audit_rollouts(
        {video_id: rollout},
        {video_id: evidence or []},
        min_run_chunks=min_run_chunks,
        drift_threshold=drift_threshold,
        top_k=20,
    )
    attempted = 0
    repaired = 0
    rejected = 0
    for item in rollout.get("thinks") or []:
        if item.get("repair"):
            repaired += 1
    # The per-think list stores only accepted repairs; derive attempted/rejected
    # from chunk log-like markers in the final memory snapshots is awkward, so
    # we re-scan the rollout with lightweight metadata if present.
    # This harness stores per-chunk repair signals only in the chunk logs, so
    # leave attempted/rejected for the aggregated pass based on those logs.
    summary = {
        "variant": spec.name,
        "video_id": video_id,
        "num_chunks": rollout.get("num_chunks", len(texts)),
        "repaired": repaired,
        "repair_attempted": attempted,
        "repair_rejected": rejected,
        "hard_stale_runs": audit["totals"]["hard_stale_runs"],
        "hard_stale_videos": audit["totals"]["hard_stale_videos"],
        "max_exact_run": _max_repeat_run(texts, near=False),
        "max_near_run": _max_repeat_run(texts, near=True),
        "top_stale_runs": audit["top_stale_runs"][:5],
    }
    return summary


def _collect_chunk_metrics(chunk_log_path: Path) -> Dict[Tuple[str, str], Dict[str, int]]:
    metrics: Dict[Tuple[str, str], Dict[str, int]] = {}
    if not chunk_log_path.exists():
        return metrics
    with chunk_log_path.open() as f:
        for line in f:
            obj = json.loads(line)
            key = (str(obj["variant"]), str(obj["video_id"]))
            cur = metrics.setdefault(
                key,
                {
                    "repair_attempted": 0,
                    "repair_rejected": 0,
                    "repaired": 0,
                },
            )
            if obj.get("repair_attempted"):
                cur["repair_attempted"] += 1
            if obj.get("repair_rejected"):
                cur["repair_rejected"] += 1
            if obj.get("repaired"):
                cur["repaired"] += 1
    return metrics


def aggregate_variant_summaries(
    summaries: Sequence[Dict],
    *,
    chunk_metrics: Dict[Tuple[str, str], Dict[str, int]],
) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for row in summaries:
        key = str(row["variant"])
        agg = grouped.setdefault(
            key,
            {
                "variant": key,
                "videos": 0,
                "chunks": 0,
                "repair_attempted": 0,
                "repair_rejected": 0,
                "repaired": 0,
                "hard_stale_runs": 0,
                "hard_stale_videos": 0,
                "max_exact_run": 0,
                "max_near_run": 0,
            },
        )
        agg["videos"] += 1
        agg["chunks"] += int(row.get("num_chunks", 0))
        cm = chunk_metrics.get((key, str(row["video_id"])), {})
        agg["repair_attempted"] += int(cm.get("repair_attempted", 0))
        agg["repair_rejected"] += int(cm.get("repair_rejected", 0))
        agg["repaired"] += int(cm.get("repaired", row.get("repaired", 0)))
        agg["hard_stale_runs"] += int(row.get("hard_stale_runs", 0))
        agg["hard_stale_videos"] += int(row.get("hard_stale_videos", 0))
        agg["max_exact_run"] = max(agg["max_exact_run"], int(row.get("max_exact_run", 0)))
        agg["max_near_run"] = max(agg["max_near_run"], int(row.get("max_near_run", 0)))

    ordered = sorted(
        grouped.values(),
        key=lambda x: (
            x["hard_stale_runs"],
            x["repair_rejected"],
            x["max_near_run"],
            x["max_exact_run"],
        ),
    )
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", default=VLLM_MODEL)
    parser.add_argument(
        "--batch-root",
        default="data/agent_v5/batch1",
        help="Batch root containing frames / evidence / audits.",
    )
    parser.add_argument(
        "--top-videos",
        type=int,
        default=3,
        help="Select the top repair-rejected videos from the batch chunk log.",
    )
    parser.add_argument(
        "--videos",
        default="",
        help="Comma-separated explicit video ids. Overrides --top-videos.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=40,
        help="Cap per-video rollout length for quick A/B iterations.",
    )
    parser.add_argument(
        "--min-run-chunks",
        type=int,
        default=8,
        help="Threshold for stale-run auditing on the partial rollouts.",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.55,
        help="Evidence drift threshold for stale-run auditing.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(spec.name for spec in DEFAULT_VARIANTS),
        help="Comma-separated variant names from DEFAULT_VARIANTS.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="vLLM request concurrency for the harness.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. Defaults to outputs/pass2_prompt_ab/<timestamp>.",
    )
    return parser.parse_args()


async def run_harness(args: argparse.Namespace) -> Path:
    batch_root = Path(args.batch_root)
    if not batch_root.is_absolute():
        batch_root = Path.cwd() / batch_root
    chunk_log = batch_root / "audits" / "pass2_chunks.jsonl"
    if args.videos:
        video_ids = [v.strip() for v in args.videos.split(",") if v.strip()]
    else:
        video_ids = select_repair_heavy_videos(chunk_log, top_k=args.top_videos)
    if not video_ids:
        raise RuntimeError(f"No candidate videos found from {chunk_log}")

    variants_by_name = {spec.name: spec for spec in DEFAULT_VARIANTS}
    variants: List[VariantSpec] = []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in variants_by_name:
            raise ValueError(
                f"Unknown variant {name!r}. Known: {sorted(variants_by_name)}"
            )
        variants.append(variants_by_name[name])

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path.cwd()
        / "outputs"
        / "pass2_prompt_ab"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_rollout_dir = out_dir / "rollouts"
    variant_rollout_dir.mkdir(parents=True, exist_ok=True)
    chunk_log_path = out_dir / "chunks.jsonl"
    chunk_log_path.write_text("", encoding="utf-8")

    config_blob = {
        "api_base": args.api_base,
        "model": args.model,
        "batch_root": str(batch_root),
        "video_ids": video_ids,
        "max_chunks": args.max_chunks,
        "min_run_chunks": args.min_run_chunks,
        "drift_threshold": args.drift_threshold,
        "variants": [asdict(v) for v in variants],
        "summary_tokens_max": SUMMARY_TOKENS_MAX,
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_blob, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    client = VLLMClient(
        api_base=args.api_base,
        model=args.model,
        max_concurrent=args.max_concurrent,
        timeout=5400.0,
    )

    summaries: List[Dict] = []
    for spec in variants:
        logger.info("Running variant %s", spec.name)
        for video_id in video_ids:
            frame_paths = _frame_paths_for_video(batch_root, video_id)
            if len(frame_paths) < FRAMES_PER_CHUNK:
                logger.warning("Skipping %s: not enough frames", video_id)
                continue
            num_chunks = min(len(frame_paths) // FRAMES_PER_CHUNK, args.max_chunks)
            evidence = _load_evidence(batch_root, video_id)
            rollout = await run_variant_single_video(
                spec=spec,
                video_id=video_id,
                frame_paths=frame_paths,
                num_chunks=num_chunks,
                client=client,
                evidence=evidence,
                chunk_log_path=chunk_log_path,
            )
            rollout_path = variant_rollout_dir / f"{spec.name}__{video_id}.json"
            rollout_path.write_text(
                json.dumps(rollout, ensure_ascii=False),
                encoding="utf-8",
            )
            summaries.append(
                summarize_variant_video(
                    spec=spec,
                    rollout=rollout,
                    evidence=evidence,
                    min_run_chunks=args.min_run_chunks,
                    drift_threshold=args.drift_threshold,
                )
            )

    chunk_metrics = _collect_chunk_metrics(chunk_log_path)
    for row in summaries:
        cm = chunk_metrics.get((str(row["variant"]), str(row["video_id"])), {})
        row["repair_attempted"] = int(cm.get("repair_attempted", 0))
        row["repair_rejected"] = int(cm.get("repair_rejected", 0))
        row["repaired"] = int(cm.get("repaired", row.get("repaired", 0)))

    aggregate = aggregate_variant_summaries(summaries, chunk_metrics=chunk_metrics)
    (out_dir / "summary.json").write_text(
        json.dumps({"per_video": summaries, "aggregate": aggregate}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "\t".join(
            [
                "variant",
                "videos",
                "chunks",
                "repair_attempted",
                "repair_rejected",
                "repaired",
                "hard_stale_runs",
                "max_exact_run",
                "max_near_run",
            ]
        )
    ]
    for row in aggregate:
        lines.append(
            "\t".join(
                [
                    row["variant"],
                    str(row["videos"]),
                    str(row["chunks"]),
                    str(row["repair_attempted"]),
                    str(row["repair_rejected"]),
                    str(row["repaired"]),
                    str(row["hard_stale_runs"]),
                    str(row["max_exact_run"]),
                    str(row["max_near_run"]),
                ]
            )
        )
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("A/B summary written to %s", out_dir / "summary.json")
    for row in aggregate:
        logger.info(
            "%s: stale=%d rejected=%d repaired=%d max_exact=%d max_near=%d",
            row["variant"],
            row["hard_stale_runs"],
            row["repair_rejected"],
            row["repaired"],
            row["max_exact_run"],
            row["max_near_run"],
        )
    return out_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    out_dir = asyncio.run(run_harness(args))
    print(out_dir)


if __name__ == "__main__":
    main()
