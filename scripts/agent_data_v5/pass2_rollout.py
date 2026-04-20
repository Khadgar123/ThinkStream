"""
Pass 2: Question-blind Streaming Rollout

Simulates the student model's real streaming experience WITHOUT any questions.
Generates: observations, compression decisions, proactive recalls, memory snapshots.

Key principle: Question-blind — no future question knowledge influences this pass.
Compression summaries use ONLY student observations (not teacher captions).

Processing: Sequential per video, parallel across videos.
"""

import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    COMPRESS_PROMPT,
    COMPRESS_RANGE,
    COMPRESS_THRESHOLD,
    OBSERVATION_PROMPT,
    PASS_CONFIG,
    PROACTIVE_RECALL_RATE,
    ROLLOUT_DIR,
    SUMMARY_TOKENS_MAX,
    SUMMARY_TOKENS_MIN,
    VISUAL_WINDOW_CHUNKS,
)
from .pass1_evidence import build_vision_content, get_chunk_frame_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory State
# ---------------------------------------------------------------------------


class MemoryState:
    """Tracks the student model's memory at each timestep."""

    def __init__(self):
        self.compressed_segments: List[Dict] = []   # {"time_range": [s,e], "text": "..."}
        self.recent_observations: List[Dict] = []   # {"chunk": N, "time": "X-Y", "obs": "..."}
        self.pending_questions: List[Dict] = []     # {"question": "...", "since_chunk": N}

    def snapshot(self, chunk_idx: int) -> Dict:
        """Create an immutable snapshot of current memory state."""
        return {
            "chunk_idx": chunk_idx,
            "compressed_segments": deepcopy(self.compressed_segments),
            "recent_observations": deepcopy(self.recent_observations),
            "pending_questions": deepcopy(self.pending_questions),
            "visual_window_start": max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1),
        }

    def add_observation(self, chunk_idx: int, observation: str):
        """Add a new observation to recent memory (from chunk leaving visual window)."""
        time_start = chunk_idx * AGENT_CHUNK_SEC
        time_end = time_start + AGENT_CHUNK_SEC
        self.recent_observations.append({
            "chunk": chunk_idx,
            "time": f"{int(time_start)}-{int(time_end)}",
            "obs": observation,
        })

    def should_compress(self) -> bool:
        return len(self.recent_observations) >= COMPRESS_THRESHOLD

    def compress(self, summary: Dict):
        """Execute compression: replace oldest observations with summary."""
        # Remove compressed observations
        self.recent_observations = self.recent_observations[COMPRESS_RANGE:]
        # Add summary
        self.compressed_segments.append(summary)

    def format_for_prompt(self) -> Tuple[str, str]:
        """Format memory state for model input prompt."""
        # Compressed segments
        compressed_text = ""
        for seg in self.compressed_segments:
            tr = seg["time_range"]
            compressed_text += f'<compressed time="{tr[0]}-{tr[1]}">{seg["text"]}</compressed>\n'

        # Recent observations
        obs_text = ""
        for item in self.recent_observations:
            obs_text += f'[{item["time"]}] {item["obs"]}\n'

        return compressed_text.strip(), obs_text.strip()


# ---------------------------------------------------------------------------
# Observation Generation
# ---------------------------------------------------------------------------


def build_observation_request(
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
) -> Dict:
    """Build request for 397B to generate a student observation.

    Input matches what the student model would see at this timestep:
    - Compressed memory segments
    - Recent observations (text)
    - Visual window (24 frames)
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)

    compressed_text, obs_text = memory.format_for_prompt()

    prompt = OBSERVATION_PROMPT.format(
        compressed_memory=compressed_text or "(none)",
        recent_observations=obs_text or "(none)",
        window_start=int(window_start * AGENT_CHUNK_SEC),
        window_end=int(end),
        start=int(start),
        end=int(end),
    )

    # Visual window frames
    window_frame_paths = []
    for c in range(window_start, chunk_idx + 1):
        window_frame_paths.extend(get_chunk_frame_paths(frame_paths, c))

    return {
        "messages": [{"role": "user", "content": build_vision_content(prompt, window_frame_paths)}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_observation"],
        "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
        "id": f"{video_id}_obs_{chunk_idx}",
    }


def parse_observation_result(raw: Optional[str]) -> str:
    """Parse observation output. Strip any residual formatting."""
    if raw is None:
        return "Scene continues without notable changes."

    # Strip think tags if present (shouldn't be with thinking=OFF, but safety)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    # Remove quotes if wrapped
    raw = raw.strip('"').strip("'").strip()
    # Truncate if too long (target 40-60 tokens ≈ 200-300 chars)
    if len(raw) > 400:
        raw = raw[:400].rsplit(" ", 1)[0]

    return raw


# ---------------------------------------------------------------------------
# Compression Generation
# ---------------------------------------------------------------------------


def build_compress_request(
    memory: MemoryState,
    video_id: str,
    chunk_idx: int,
) -> Dict:
    """Build compression request for the oldest 10 observations."""
    to_compress = memory.recent_observations[:COMPRESS_RANGE]

    # Format observations text
    obs_lines = [f'[{item["time"]}] {item["obs"]}' for item in to_compress]
    obs_text = "\n".join(obs_lines)

    # Determine time range
    first_time = to_compress[0]["chunk"] * AGENT_CHUNK_SEC
    last_time = to_compress[-1]["chunk"] * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC

    # Adaptive target length based on content complexity
    target_length = estimate_summary_length(to_compress)

    prompt = COMPRESS_PROMPT.format(
        observations_text=obs_text,
        target_length=target_length,
        start=int(first_time),
        end=int(last_time),
    )

    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_compress"],
        "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
        "id": f"{video_id}_compress_{chunk_idx}",
        "_meta": {"time_range": [int(first_time), int(last_time)]},
    }


def parse_compress_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse compression summary output."""
    default = {
        "time_range": meta["time_range"],
        "text": "Observations recorded during this period.",
        "parse_success": False,
    }

    if raw is None:
        return default

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
        return {
            "time_range": parsed.get("time_range", meta["time_range"]),
            "text": parsed.get("text", ""),
            "parse_success": True,
        }
    except (json.JSONDecodeError, ValueError):
        # Try to extract JSON
        start = raw.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(raw[start:i + 1])
                            return {
                                "time_range": parsed.get("time_range", meta["time_range"]),
                                "text": parsed.get("text", ""),
                                "parse_success": True,
                            }
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

    default["_raw"] = raw[:300]
    return default


def estimate_summary_length(observations: List[Dict]) -> int:
    """Estimate appropriate summary length based on content complexity."""
    text = " ".join(item["obs"] for item in observations)
    # Count unique entity-like words (capitalized or specific patterns)
    words = set(w for w in text.split() if len(w) > 0)
    n_entities = sum(1 for w in words if len(w) > 0 and (w[0].isupper() or "_" in w))
    has_numbers = any(c.isdigit() for c in text)

    base = SUMMARY_TOKENS_MIN  # 100
    base += min(n_entities * 8, 60)
    base += 20 if has_numbers else 0

    return min(base, SUMMARY_TOKENS_MAX)


# ---------------------------------------------------------------------------
# Main Rollout
# ---------------------------------------------------------------------------


async def run_pass2_single_video(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,
) -> Dict:
    """Run question-blind streaming rollout for a single video.

    Returns complete timeline with observations, compressions, and snapshots.
    """
    memory = MemoryState()
    observations = []         # All observations generated
    compression_events = []   # All compression events
    snapshots = {}            # Memory state at each timestep
    proactive_recalls = []    # Proactive recall events

    for chunk_idx in range(num_chunks):
        # --- Snapshot BEFORE this timestep's action ---
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)

        # --- Generate observation ---
        request = build_observation_request(chunk_idx, frame_paths, memory, video_id)
        raw = await client._call_one(
            messages=request["messages"],
            max_tokens=request["max_tokens"],
            temperature=request["temperature"],
            request_id=request["id"],
        )
        observation = parse_observation_result(raw)
        observations.append({
            "chunk_idx": chunk_idx,
            "time": [chunk_idx * AGENT_CHUNK_SEC, (chunk_idx + 1) * AGENT_CHUNK_SEC],
            "observation": observation,
        })

        # --- Update memory: chunk leaving visual window enters text memory ---
        leaving_chunk = chunk_idx - VISUAL_WINDOW_CHUNKS
        if leaving_chunk >= 0:
            memory.add_observation(leaving_chunk, observations[leaving_chunk]["observation"])

        # --- Check compression trigger ---
        if memory.should_compress():
            comp_request = build_compress_request(memory, video_id, chunk_idx)
            comp_raw = await client._call_one(
                messages=comp_request["messages"],
                max_tokens=comp_request["max_tokens"],
                temperature=comp_request["temperature"],
                request_id=comp_request["id"],
            )
            summary = parse_compress_result(comp_raw, comp_request["_meta"])
            memory.compress(summary)

            compression_events.append({
                "trigger_chunk": chunk_idx,
                "summary": summary,
                "compressed_obs_chunks": list(range(
                    memory.compressed_segments[-1]["time_range"][0] // AGENT_CHUNK_SEC,
                    memory.compressed_segments[-1]["time_range"][1] // AGENT_CHUNK_SEC,
                )),
            })
            logger.debug(f"  [{video_id}] Compress at chunk {chunk_idx}: {summary['time_range']}")

        # --- Proactive recall (low frequency, ~5%) ---
        # Only after we have compressed memories to potentially recall from
        if (memory.compressed_segments
                and random.random() < PROACTIVE_RECALL_RATE
                and chunk_idx > VISUAL_WINDOW_CHUNKS + COMPRESS_THRESHOLD):
            proactive_recalls.append({
                "chunk_idx": chunk_idx,
                "reason": "proactive_entity_connection",
            })

        # Progress logging
        if (chunk_idx + 1) % 10 == 0:
            logger.info(
                f"  [{video_id}] Rollout: {chunk_idx+1}/{num_chunks} "
                f"(memory: {len(memory.recent_observations)} obs, "
                f"{len(memory.compressed_segments)} compressed)"
            )

    return {
        "video_id": video_id,
        "num_chunks": num_chunks,
        "observations": observations,
        "compression_events": compression_events,
        "proactive_recalls": proactive_recalls,
        "snapshots": snapshots,
        "final_memory": memory.snapshot(num_chunks),
    }


def save_rollout(video_id: str, rollout: Dict, output_dir: Path = ROLLOUT_DIR):
    """Save rollout results for one video.

    Note: snapshot keys are converted to strings during JSON serialization.
    load_rollout handles this by normalizing keys back to int.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rollout, f, ensure_ascii=False)


def load_rollout(video_id: str, rollout_dir: Path = ROLLOUT_DIR) -> Optional[Dict]:
    """Load cached rollout if available.

    Normalizes snapshot keys from str (JSON artifact) back to int.
    """
    path = rollout_dir / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Normalize snapshot keys to int
    if "snapshots" in data:
        data["snapshots"] = {int(k): v for k, v in data["snapshots"].items()}
    return data
