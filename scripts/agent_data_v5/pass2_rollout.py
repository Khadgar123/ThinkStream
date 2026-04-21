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
    COMPRESS_RANGE_MAX,
    COMPRESS_RANGE_MIN,
    COMPRESS_TOKEN_THRESHOLD,
    OBSERVATION_PROMPT,
    RECENT_THINKS_TOKEN_BUDGET,
    get_tokenizer,
    PASS_CONFIG,
    PROACTIVE_RECALL_RATE,
    ROLLOUT_DIR,
    MAX_COMPRESSED_SEGMENTS,
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
    """Tracks the student model's text memory at each timestep.

    Key design: text memory covers LONGER time than visual window.
    - Think is generated every chunk and IMMEDIATELY enters recent_thinks.
    - Visual window only holds the last 12 chunks of frames.
    - So text memory and visual window OVERLAP for recent chunks,
      but text memory extends further back in time.

    The model sees: compressed_segments + recent_thinks + pending_questions
    System-side: _retrieval_archive keeps all historical thinks for recall.
    """

    def __init__(self):
        self.compressed_segments: List[Dict] = []   # {"time_range": [s,e], "text": "..."}
        self.recent_thinks: List[Dict] = []         # {"chunk": N, "time": "X-Y", "text": "..."}
        self.pending_questions: List[Dict] = []     # {"question": "...", "since_chunk": N}
        self._retrieval_archive: List[Dict] = []    # system-side: all past thinks

    @property
    def retrieval_archive(self) -> List[Dict]:
        return self._retrieval_archive

    # Keep backward compat alias
    @property
    def recent_observations(self) -> List[Dict]:
        return self.recent_thinks

    def snapshot(self, chunk_idx: int) -> Dict:
        """Snapshot of what the model sees (no archive)."""
        return {
            "chunk_idx": chunk_idx,
            "compressed_segments": deepcopy(self.compressed_segments),
            "recent_thinks": deepcopy(self.recent_thinks),
            "pending_questions": deepcopy(self.pending_questions),
            "visual_window_start": max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1),
        }

    def add_think(self, chunk_idx: int, think_text: str):
        """Add think to memory IMMEDIATELY (not delayed until leaving window).

        Text memory covers longer time than visual window — think is added
        right after generation, even if the chunk is still in visual window.
        """
        time_start = chunk_idx * AGENT_CHUNK_SEC
        time_end = time_start + AGENT_CHUNK_SEC
        item = {
            "chunk": chunk_idx,
            "time": f"{int(time_start)}-{int(time_end)}",
            "text": think_text,
        }
        self.recent_thinks.append(item)
        self._retrieval_archive.append(item)

    def count_recent_tokens(self) -> int:
        """Count total tokens in recent_thinks using student tokenizer.

        Uses real tokenizer when available (precise), falls back to
        chars/4 estimate otherwise.
        """
        tokenizer = get_tokenizer()
        total = 0
        for item in self.recent_thinks:
            text = item.get("text", "")
            if tokenizer:
                total += len(tokenizer.encode(text, add_special_tokens=False))
            else:
                total += len(text) // 4  # fallback: ~4 chars/token
        return total

    def should_compress(self) -> bool:
        """Trigger compression when recent_thinks reach 80% of token budget."""
        return self.count_recent_tokens() >= COMPRESS_TOKEN_THRESHOLD

    def compress(self, summary: Dict, compressed_chunks: Optional[List[int]] = None):
        """Replace specified thinks with summary in model context.

        Args:
            summary: The compressed summary dict.
            compressed_chunks: List of chunk indices that were compressed.
                If None, removes the first COMPRESS_RANGE items (backward compat).

        Raw thinks stay in _retrieval_archive for recall.
        Merges oldest two segments when over limit.
        """
        if compressed_chunks is not None:
            chunk_set = set(compressed_chunks)
            self.recent_thinks = [t for t in self.recent_thinks if t["chunk"] not in chunk_set]
        else:
            self.recent_thinks = self.recent_thinks[COMPRESS_RANGE:]
        self.compressed_segments.append(summary)
        while len(self.compressed_segments) > MAX_COMPRESSED_SEGMENTS:
            seg_a = self.compressed_segments.pop(0)
            seg_b = self.compressed_segments.pop(0)
            combined = f'{seg_a["text"]} {seg_b["text"]}'
            # Truncate merged text to ~200 tokens (not words) using tokenizer
            tokenizer = get_tokenizer()
            if tokenizer:
                ids = tokenizer.encode(combined, add_special_tokens=False)
                if len(ids) > 200:
                    combined = tokenizer.decode(ids[:200])
            else:
                words = combined.split()
                if len(words) > 200:
                    combined = " ".join(words[:200])
            merged = {
                "time_range": [seg_a["time_range"][0], seg_b["time_range"][1]],
                "text": combined,
                "merged": True,
                "merge_level": max(
                    seg_a.get("merge_level", 1),
                    seg_b.get("merge_level", 1),
                ) + 1,
            }
            self.compressed_segments.insert(0, merged)

    def format_for_prompt(self) -> Tuple[str, str]:
        """Format memory state for model input prompt."""
        compressed_text = ""
        for seg in self.compressed_segments:
            tr = seg["time_range"]
            compressed_text += f'<compressed time="{tr[0]}-{tr[1]}">{seg["text"]}</compressed>\n'

        thinks_text = ""
        for item in self.recent_thinks:
            thinks_text += f'[{item["time"]}] {item["text"]}\n'

        return compressed_text.strip(), thinks_text.strip()


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
    """Build compression request using teacher-chosen optimal range (legacy)."""
    to_compress = choose_optimal_compress_range(memory.recent_thinks, memory.pending_questions)

    # Format thinks text
    obs_lines = [f'[{item["time"]}] {item["text"]}' for item in to_compress]
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
        "_meta": {
            "time_range": [int(first_time), int(last_time)],
            "chunks": [item["chunk"] for item in to_compress],
        },
    }


def build_compress_request_from_thinks(
    pre_action_thinks: List[Dict],
    memory: MemoryState,
    video_id: str,
    chunk_idx: int,
) -> Dict:
    """Build compression request using ONLY pre-action thinks.

    Key constraint: compress range must only cover thinks that were in the
    model's INPUT (pre-action snapshot), not including the current think
    that was just generated. This ensures the model can learn to compress
    based on what it actually saw.
    """
    to_compress = choose_optimal_compress_range(pre_action_thinks, memory.pending_questions)

    obs_lines = [f'[{item["time"]}] {item.get("text", item.get("obs", ""))}' for item in to_compress]
    obs_text = "\n".join(obs_lines)

    first_time = to_compress[0]["chunk"] * AGENT_CHUNK_SEC
    last_time = to_compress[-1]["chunk"] * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC

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
        "_meta": {
            "time_range": [int(first_time), int(last_time)],
            "chunks": [item["chunk"] for item in to_compress],
        },
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
            "time_range": meta["time_range"],  # Always use real meta, not model output
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
                                "time_range": meta["time_range"],
                                "text": parsed.get("text", ""),
                                "parse_success": True,
                            }
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

    default["_raw"] = raw[:300]
    return default


def score_range_for_compression(
    thinks: List[Dict],
    all_recent: List[Dict],
    pending_questions: Optional[List[Dict]] = None,
) -> float:
    """Score a candidate range for compression. LOWER = better to compress.

    Multi-dimensional scoring per 记忆结构修改.txt:
    - importance_lost: entities, numbers, OCR, state changes (higher = worse)
    - pending_overlap_penalty: overlap with pending question content (higher = worse)
    - event_boundary_penalty: range doesn't start/end at low-change points (higher = worse)
    - token_saving_gain: more tokens saved = better (lower score)
    - reconstructability_bonus: repetitive/simple content is easy to compress (lower score)
    """
    text = " ".join(item.get("text", item.get("obs", "")) for item in thinks)
    words = text.split()
    n_words = len(words)

    # --- importance_lost ---
    unique_words = set(w.lower() for w in words)
    n_entities = sum(1 for w in set(words) if w and (w[0].isupper() or "_" in w))
    has_numbers = any(c.isdigit() for c in text)
    has_ocr = any(kw in text.lower() for kw in ["ocr", "text", "label", "sign", "read", "display"])
    importance = n_entities * 2.0 + (8.0 if has_numbers else 0) + (8.0 if has_ocr else 0)

    # --- pending_overlap_penalty ---
    pending_penalty = 0.0
    if pending_questions:
        for pq in pending_questions:
            q_words = set(pq.get("question", "").lower().split())
            overlap = len(unique_words & q_words)
            if overlap > 0:
                pending_penalty += overlap * 10.0  # Heavy penalty

    # --- event_boundary_penalty ---
    # Check if first/last thinks share entities with neighbors outside range
    range_chunks = set(t["chunk"] for t in thinks)
    boundary_penalty = 0.0
    for t in all_recent:
        if t["chunk"] not in range_chunks:
            neighbor_text = t.get("text", t.get("obs", "")).lower()
            # Check overlap with boundary thinks
            if thinks:
                first_text = thinks[0].get("text", "").lower()
                last_text = thinks[-1].get("text", "").lower()
                first_words = set(first_text.split())
                last_words = set(last_text.split())
                neighbor_words = set(neighbor_text.split())
                if first_words & neighbor_words - {"the", "a", "on", "in", "at"}:
                    boundary_penalty += 2.0
                if last_words & neighbor_words - {"the", "a", "on", "in", "at"}:
                    boundary_penalty += 2.0

    # --- token_saving_gain (negative = good, saves more) ---
    # Use tokenizer for precise token count when available
    tokenizer = get_tokenizer()
    if tokenizer:
        n_tokens = sum(
            len(tokenizer.encode(item.get("text", item.get("obs", "")), add_special_tokens=False))
            for item in thinks
        )
    else:
        n_tokens = len(text) // 4  # fallback: ~4 chars/token
    token_saving = -n_tokens * 0.3  # More tokens → more savings → lower score

    # --- reconstructability_bonus (repetitive content is easy to compress) ---
    unique_ratio = len(unique_words) / max(n_words, 1)
    reconstructability = -5.0 if unique_ratio < 0.5 else 0.0  # Low diversity = easy

    score = importance + pending_penalty + boundary_penalty + token_saving + reconstructability
    return score


def choose_optimal_compress_range(
    recent_thinks: List[Dict],
    pending_questions: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Choose the contiguous range that is BEST to compress (greedy optimal).

    Teacher-guided: evaluates all valid contiguous ranges, scores each by
    multi-dimensional criteria (importance loss, pending overlap, event
    boundary, token saving, reconstructability). Picks lowest score.

    This is the gold range for C1 training.
    """
    n = len(recent_thinks)
    best_range = None
    best_score = float("inf")

    for size in range(COMPRESS_RANGE_MIN, min(COMPRESS_RANGE_MAX + 1, n + 1)):
        for start in range(0, n - size + 1):
            candidate = recent_thinks[start:start + size]
            score = score_range_for_compression(
                candidate, recent_thinks, pending_questions
            )
            if score < best_score:
                best_score = score
                best_range = candidate

    return best_range if best_range else recent_thinks[:COMPRESS_RANGE_MIN]


def estimate_summary_length(observations: List[Dict]) -> int:
    """Estimate appropriate summary length based on content complexity."""
    text = " ".join(item.get("text", item.get("obs", "")) for item in observations)
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

    Key: text memory (thinks) covers LONGER time than visual window.
    Think is generated and enters memory IMMEDIATELY — no delay until
    leaving the visual window.
    """
    memory = MemoryState()
    thinks = []               # All thinks generated (full timeline)
    compression_events = []   # All compression events
    snapshots = {}            # Memory state at each timestep
    proactive_recalls = []    # Proactive recall events

    for chunk_idx in range(num_chunks):
        # --- 1. Snapshot BEFORE this step's think (pre-action state) ---
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)

        # --- 2. Generate think for current chunk ---
        request = build_observation_request(chunk_idx, frame_paths, memory, video_id)
        raw = await client._call_one(
            messages=request["messages"],
            max_tokens=request["max_tokens"],
            temperature=request["temperature"],
            request_id=request["id"],
        )
        think_text = parse_observation_result(raw)
        thinks.append({
            "chunk_idx": chunk_idx,
            "time": [chunk_idx * AGENT_CHUNK_SEC, (chunk_idx + 1) * AGENT_CHUNK_SEC],
            "think": think_text,
        })

        # --- 3. Think enters memory IMMEDIATELY (not delayed) ---
        memory.add_think(chunk_idx, think_text)

        # --- 4. Check compression trigger (80% of capacity) ---
        if memory.should_compress():
            # IMPORTANT: compress range must only cover thinks that were in
            # the PRE-ACTION snapshot (i.e. before current think was added).
            # We select range from pre-action thinks, not from current memory.
            pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]
            comp_request = build_compress_request_from_thinks(
                pre_action_thinks, memory, video_id, chunk_idx
            )
            comp_raw = await client._call_one(
                messages=comp_request["messages"],
                max_tokens=comp_request["max_tokens"],
                temperature=comp_request["temperature"],
                request_id=comp_request["id"],
            )
            summary = parse_compress_result(comp_raw, comp_request["_meta"])
            real_chunks = comp_request["_meta"]["chunks"]
            memory.compress(summary, compressed_chunks=real_chunks)

            compression_events.append({
                "trigger_chunk": chunk_idx,
                "summary": summary,
                "compressed_thinks_chunks": real_chunks,
            })
            logger.debug(f"  [{video_id}] Compress at chunk {chunk_idx}: {summary['time_range']}")

        # --- Proactive recall (low frequency, ~5%) ---
        # Only after enough time has passed to have compressed memories
        if (memory.compressed_segments
                and random.random() < PROACTIVE_RECALL_RATE
                and chunk_idx > VISUAL_WINDOW_CHUNKS + 10):
            proactive_recalls.append({
                "chunk_idx": chunk_idx,
                "reason": "proactive_entity_connection",
            })

        # Progress logging
        if (chunk_idx + 1) % 10 == 0:
            logger.info(
                f"  [{video_id}] Rollout: {chunk_idx+1}/{num_chunks} "
                f"(memory: {len(memory.recent_thinks)} thinks, "
                f"{len(memory.compressed_segments)} compressed)"
            )

    return {
        "video_id": video_id,
        "num_chunks": num_chunks,
        "thinks": thinks,
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
