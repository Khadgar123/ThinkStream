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


def _build_visual_window(chunk_idx: int, num_chunks: int, video_path: str) -> Dict:
    """Build visual_window structure from chunk index."""
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    video_start = window_start * AGENT_CHUNK_SEC
    video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
    n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

    return {
        "video_start": video_start,
        "video_end": video_end,
        "frames": n_frames,
    }


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


def render_sample(
    sample: Dict,
    rollout: Dict,
    video_path: str,
    video_id: str,
) -> Dict:
    """Render a 3-C sample into an SFT-ready training sample.

    Combines:
    - sample's output, queries, user_input, recall_result (from 3-C)
    - rollout's snapshot at chunk_idx (from Pass 2)
    - system prompt (from agent_protocol)
    - visual_window structure (computed from chunk_idx)

    Returns a complete sample with `input` + `output` fields.
    """
    chunk_idx = sample["chunk_idx"]
    prompt_type = sample.get("prompt_type", "SYSTEM_PROMPT")
    num_chunks = rollout["num_chunks"]

    # Get snapshot for this chunk
    snapshots = rollout["snapshots"]
    snapshot = snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx)) or {}

    # Build input structure
    inp = {
        "system": _get_system_prompt(prompt_type),
        "visual_window": _build_visual_window(chunk_idx, num_chunks, video_path),
        "memory": _build_memory_from_snapshot(snapshot),
        "queries": _build_queries_input(sample.get("queries", [])),
        "user_input": sample.get("user_input", ""),
        "recall_result": sample.get("recall_result"),
    }

    # For compress samples, add compress_trigger to user_input
    if sample.get("action") == "compress":
        # Find the compression event for this chunk
        for event in rollout.get("compression_events", []):
            if event.get("trigger_chunk") == chunk_idx:
                cr = event.get("compressed_thinks_chunks", [])
                if cr:
                    time_start = min(cr) * AGENT_CHUNK_SEC
                    time_end = (max(cr) + 1) * AGENT_CHUNK_SEC
                    inp["user_input"] = f'<compress_trigger range="{time_start}-{time_end}"/>'
                break

    # Build complete SFT sample
    rendered = {
        # Core fields for SFT data_processor
        "input": inp,
        "output": sample["output"],
        "video_path": video_path,
        "video_id": video_id,
        "chunk_idx": chunk_idx,

        # Metadata for phase assignment and verification
        "sample_type": sample.get("sample_type", "silent"),
        "action": sample.get("action", "silent"),
        "prompt_type": prompt_type,
        "sequence_type": sample.get("sequence_type", ""),
        "trajectory_id": sample.get("trajectory_id", ""),
        "card_id": sample.get("card_id", ""),
    }

    return rendered


def render_trajectory(
    trajectory_samples: List[Dict],
    rollout: Dict,
    video_path: str,
    video_id: str,
) -> List[Dict]:
    """Render all samples in a trajectory into SFT-ready format.

    Handles the queries_state override: for each sample, the queries
    from the 3-C fork generation take precedence over the rollout
    snapshot's queries (which is always empty in question-blind rollout).
    """
    rendered = []
    for sample in trajectory_samples:
        try:
            r = render_sample(sample, rollout, video_path, video_id)
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
) -> List[Dict]:
    """Render all samples for a video, grouped by trajectory.

    Each trajectory's samples share a queries_state evolution.
    Base samples inherit queries_state from their trajectory.
    """
    # Group by trajectory_id
    by_traj = {}
    for s in all_samples:
        tid = s.get("trajectory_id", "no_traj")
        by_traj.setdefault(tid, []).append(s)

    rendered = []
    for tid, traj_samples in by_traj.items():
        traj_rendered = render_trajectory(traj_samples, rollout, video_path, video_id)
        rendered.extend(traj_rendered)

    return rendered
