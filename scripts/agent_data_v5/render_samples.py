"""
Render Pass: Convert 3-C samples into SFT-ready training samples.

Takes: 3-C fork+base samples + Pass 2 rollout snapshots
Produces: Complete SFT samples with input + output fields

This is the bridge between data construction (Pass 3) and SFT training.
Each sample gets a full `input` structure that matches what
`data_processor.py:build_per_timestep_messages` expects.

Called after Pass 4 verify, before writing final JSONL.
"""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    VISUAL_WINDOW_CHUNKS,
    FRAMES_PER_CHUNK,
)

logger = logging.getLogger(__name__)

# System prompt — canonical copy in agent_protocol.py
# Import to guarantee train/inference identity
try:
    from thinkstream.data.agent_protocol import SYSTEM_PROMPT
except ImportError:
    SYSTEM_PROMPT = "You are a streaming video agent..."

# Post-recall prompt (2-action: silent / response)
POST_RECALL_SYSTEM_PROMPT = (
    "The system retrieved past observations based on your recall query. "
    "Review the recall_result and decide:\n\n"
    "1) <think>analysis</think><action>response</action><response>answer</response>\n"
    "   If the result contains enough evidence to answer.\n\n"
    "2) <think>analysis</think><action>silent</action>\n"
    "   If the result is insufficient or irrelevant.\n\n"
    "Think: analyze whether the recall result answers the question (20-40 tokens)."
)

# Compress prompt
COMPRESS_SYSTEM_PROMPT = (
    "System: memory token budget exceeded. Compress the specified thinks into a summary.\n\n"
    "Output: <summary>{\"time_range\":[s,e],\"text\":\"...\"}</summary>\n\n"
    "Rules: retain ALL entity names, visual attributes, OCR text, numbers, and state changes."
)


def _get_system_prompt(prompt_type: str) -> str:
    """Select system prompt by sample's prompt_type."""
    if prompt_type == "POST_RECALL_PROMPT":
        return POST_RECALL_SYSTEM_PROMPT
    elif prompt_type == "COMPRESS_PROMPT":
        return COMPRESS_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _build_visual_window(
    chunk_idx: int, num_chunks: int, video_path: str,
    frame_dir: str = None,
) -> Dict:
    """Build visual_window structure from chunk index.

    If frame_dir is provided (pre-extracted frames), includes frame_paths
    for fast training I/O. Otherwise SFT data_processor falls back to
    video_path online decoding.
    """
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    vw = {
        "video_start": video_start,
        "video_end": video_end,
        "frames": n_frames,
    }

    # If pre-extracted frames exist, add frame_paths for fast I/O
    if frame_dir:
        paths = []
        for ci in range(window_start, chunk_idx + 1):
            for fi in range(FRAMES_PER_CHUNK):
                p = Path(frame_dir) / f"chunk_{ci:04d}_f{fi}.jpg"
                if p.exists():
                    paths.append(str(p))
        if paths:
            vw["frame_paths"] = paths

    return vw


def _build_memory_from_snapshot(snapshot: Dict) -> Dict:
    """Convert rollout snapshot to memory structure for SFT input."""
    memory = {
        "compressed_segments": [],
        "recent_thinks": [],
    }

    # Compressed segments from snapshot
    for seg in snapshot.get("compressed_segments", []):
        memory["compressed_segments"].append({
            "time_range": seg["time_range"],
            "text": seg["text"],
        })

    # Recent thinks from snapshot
    for item in snapshot.get("recent_thinks", []):
        if isinstance(item, dict):
            time_str = item.get("time", "")
            text = item.get("text", item.get("obs", ""))
            memory["recent_thinks"].append(f"[{time_str}] {text}")
        elif isinstance(item, str):
            memory["recent_thinks"].append(item)

    return memory


def _build_queries_input(queries_state: List[Dict]) -> List[Dict]:
    """Convert queries_state to the format expected by SFT input."""
    result = []
    for q in queries_state:
        result.append({
            "question": q.get("question", ""),
            "answers": q.get("answers", []),
        })
    return result


def _build_recalled_frames(
    recall_result: Optional[Dict],
    all_frame_paths: Optional[List[str]],
) -> Optional[Dict]:
    """Build the `recalled_frames` input zone for recall_response samples.

    Mirrors the inference-time logic in agent_loop.step (recall branch):
    given returned_chunks from a successful retrieval, derive the
    contiguous time_range, frame count, and per-chunk frame_paths so the
    SFT sample renders <recalled_frames> + actual video frames — not
    text-only. Without this, SFT trains on the recall_result text alone
    and inference's frame injection becomes OOD.

    Returns None for failure / distractor / empty results — those
    samples carry only <recall_result> text.
    """
    if not recall_result or recall_result.get("source") != "historical_frames":
        return None
    chunks = recall_result.get("returned_chunks") or []
    if not chunks:
        return None
    t_start = min(chunks) * AGENT_CHUNK_SEC
    t_end = (max(chunks) + 1) * AGENT_CHUNK_SEC
    rf = {
        "time_range": [int(t_start), int(t_end)],
        "n_frames": len(chunks) * FRAMES_PER_CHUNK,
        "source": "historical_frames",
    }
    if all_frame_paths:
        from .pass1a_evidence import get_chunk_frame_paths
        paths = []
        for c in chunks:
            paths.extend(get_chunk_frame_paths(all_frame_paths, c))
        if paths:
            rf["frame_paths"] = paths
    return rf


def render_sample(
    sample: Dict,
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> Dict:
    """Render a 3-C sample into an SFT-ready training sample.

    Combines:
    - sample's output, queries, user_input, recall_result (from 3-C)
    - rollout's snapshot at chunk_idx (from Pass 2)
    - system prompt (from agent_protocol)
    - visual_window structure (computed from chunk_idx)
    - metadata with gold_action/gold_answer (from cards_map)

    `all_frame_paths` is the per-video flat frame list (1fps extracted
    earlier in the pipeline). When provided, recall_response samples get
    `recalled_frames.frame_paths` populated so the SFT loader feeds actual
    historical frames into the model — matching what inference does. If
    omitted, recall_response samples render text-only recall (legacy
    behaviour, retained for back-compat with callers that don't have the
    frame list).

    Returns a complete sample with `input` + `output` + `metadata` fields.
    """
    chunk_idx = sample["chunk_idx"]
    prompt_type = sample.get("prompt_type", "SYSTEM_PROMPT")
    num_chunks = rollout["num_chunks"]

    # Get snapshot for this chunk
    snapshots = rollout["snapshots"]
    snapshot = snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx)) or {}

    # Build input structure
    recall_result = sample.get("recall_result")
    inp = {
        "system": _get_system_prompt(prompt_type),
        "visual_window": _build_visual_window(chunk_idx, num_chunks, video_path),
        "memory": _build_memory_from_snapshot(snapshot),
        "queries": _build_queries_input(sample.get("queries", [])),
        "user_input": sample.get("user_input", ""),
        "recall_result": recall_result,
    }
    rf = _build_recalled_frames(recall_result, all_frame_paths)
    if rf is not None:
        inp["recalled_frames"] = rf

    # For compress samples, add compress_trigger to user_input AND
    # remember the gold compressed-chunks set so RL/eval can score the
    # model's <summary> time_range against teacher's choice.
    gold_compress_chunks: List[int] = []
    if sample.get("action") == "compress":
        for event in rollout.get("compression_events", []):
            if event.get("trigger_chunk") == chunk_idx:
                cr = event.get("compressed_thinks_chunks", [])
                if cr:
                    time_start = min(cr) * AGENT_CHUNK_SEC
                    time_end = (max(cr) + 1) * AGENT_CHUNK_SEC
                    inp["user_input"] = f'<compress_trigger range="{time_start}-{time_end}"/>'
                    gold_compress_chunks = sorted(int(c) for c in cr)
                break

    # Build metadata for RL reward computation.
    #
    # Contract (silent failures here ⇒ Pass4 + GRPO get neutered):
    #   - if sample has a card_id, the card MUST be in cards_map.
    #     A missing lookup means cards_map was passed empty/wrong upstream;
    #     fail loudly rather than emit an empty-metadata sample.
    #   - silent-only samples (no card_id) legitimately have empty metadata.
    card_id = sample.get("card_id", "")
    if card_id:
        if not cards_map or card_id not in cards_map:
            raise KeyError(
                f"[{video_id}] sample chunk={chunk_idx} references card_id="
                f"{card_id!r} but it is missing from cards_map "
                f"(cards_map size={len(cards_map or {})}). Render before Pass4 "
                f"requires fully-populated cards_map."
            )
        card = cards_map[card_id]
    else:
        card = {}
    metadata = {
        "gold_action": sample.get("action", "silent"),
        "gold_answer": card.get("canonical_answer", ""),
        "answer_form": card.get("answer_form", ""),
        "family": card.get("family", ""),
        "availability": sample.get("sequence_type", ""),
        "support_chunks": card.get("support_chunks", []),
        # Gold compressed-chunks (for compress samples) — empty list for
        # non-compress samples. Used by streaming-eval / RL to score
        # the model's <summary> time_range vs teacher's policy choice.
        "gold_compress_chunks": gold_compress_chunks,
    }

    # Build complete SFT sample
    rendered = {
        # Core fields for SFT data_processor
        "input": inp,
        "output": sample["output"],
        "video_path": video_path,
        "video_id": video_id,
        "chunk_idx": chunk_idx,

        # Phase assignment
        "sample_type": sample.get("sample_type", "silent"),
        "action": sample.get("action", "silent"),
        "prompt_type": prompt_type,
        "sequence_type": sample.get("sequence_type", ""),
        "trajectory_id": sample.get("trajectory_id", ""),
        "card_id": card_id,

        # RL reward fields
        "metadata": metadata,
    }

    # Propagate base_role for trajectory-aware loss weighting
    if "base_role" in sample:
        rendered["base_role"] = sample["base_role"]

    return rendered


def render_trajectory(
    trajectory_samples: List[Dict],
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Render all samples in a trajectory into SFT-ready format."""
    rendered = []
    for sample in trajectory_samples:
        try:
            r = render_sample(sample, rollout, video_path, video_id,
                              cards_map, all_frame_paths=all_frame_paths)
            rendered.append(r)
        except Exception as e:
            logger.warning(
                f"[{video_id}] render failed for chunk {sample.get('chunk_idx')}: {e}"
            )
    return rendered


def render_video_samples(
    all_samples: List[Dict],
    rollout: Dict,
    video_path: str,
    video_id: str,
    cards_map: Dict[str, Dict] = None,
    all_frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Render all samples for a video, grouped by trajectory."""
    by_traj = {}
    for s in all_samples:
        tid = s.get("trajectory_id", "no_traj")
        by_traj.setdefault(tid, []).append(s)

    rendered = []
    for tid, traj_samples in by_traj.items():
        traj_rendered = render_trajectory(
            traj_samples, rollout, video_path, video_id, cards_map,
            all_frame_paths=all_frame_paths)
        rendered.extend(traj_rendered)

    return rendered
