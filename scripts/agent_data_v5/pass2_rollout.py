"""
Pass 2: Question-blind Streaming Rollout

Simulates the student model's real streaming experience WITHOUT any questions.
Generates: observations, compression decisions, memory snapshots.

Key principle: Question-blind — no future question knowledge influences this pass.
Compression summaries use ONLY student observations (not teacher captions).
Compression is triggered BETWEEN timesteps (separate prompt, does not occupy a timestep).

Processing: Sequential per video, parallel across videos.
"""

import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    COMPRESS_PROMPT,
    COMPRESS_RANGE_MAX,
    COMPRESS_RANGE_MIN,
    COMPRESS_TOKEN_THRESHOLD,
    COMPRESS_HYSTERESIS_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    OBSERVATION_PROMPT,
    get_tokenizer,
    PASS_CONFIG,
    ROLLOUT_DIR,
    MAX_COMPRESSED_SEGMENTS,
    SUMMARY_TOKENS_MAX,
    SUMMARY_TOKENS_MIN,
    VISUAL_WINDOW_CHUNKS,
)
from .pass1a_evidence import build_vision_content, get_chunk_frame_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory State
# ---------------------------------------------------------------------------


class MemoryState:
    """Tracks the student model's text memory at each timestep.

    v8.0: Unified timeline — summary and thinks in one list, chronological order.
    Compression = in-place replacement (selected thinks → summary, same position).
    No separate compressed_segments / recent_thinks zones.

    Queries managed in separate <queries> zone, independent of memory.
    """

    def __init__(self):
        # Unified timeline: mix of thinks and summaries, chronological order
        # Think item: {"type": "think", "chunk": N, "time": "X-Y", "text": "..."}
        # Summary item: {"type": "summary", "time_range": [s,e], "text": "..."}
        self.timeline: List[Dict] = []
        self._retrieval_archive: List[Dict] = []  # system-side: all past thinks (never compressed)

    @property
    def retrieval_archive(self) -> List[Dict]:
        return self._retrieval_archive

    # Backward compat: downstream code may access these
    @property
    def compressed_segments(self) -> List[Dict]:
        return [item for item in self.timeline if item.get("type") == "summary"]

    @property
    def recent_thinks(self) -> List[Dict]:
        return [item for item in self.timeline if item.get("type") == "think"]

    def snapshot(self, chunk_idx: int) -> Dict:
        """Snapshot of what the model sees (no archive, no queries)."""
        return {
            "chunk_idx": chunk_idx,
            "timeline": deepcopy(self.timeline),
            # Backward compat fields (derived from timeline)
            "compressed_segments": deepcopy(self.compressed_segments),
            "recent_thinks": deepcopy(self.recent_thinks),
            "visual_window_start": max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1),
        }

    def add_think(self, chunk_idx: int, think_text: str):
        """Append think to timeline."""
        time_start = chunk_idx * AGENT_CHUNK_SEC
        time_end = time_start + AGENT_CHUNK_SEC
        item = {
            "type": "think",
            "chunk": chunk_idx,
            "time": f"{int(time_start)}-{int(time_end)}",
            "text": think_text,
        }
        self.timeline.append(item)
        self._retrieval_archive.append(item)

    def count_tokens(self) -> int:
        """Count total tokens in timeline (thinks + summaries)."""
        tokenizer = get_tokenizer()
        total = 0
        for item in self.timeline:
            text = item.get("text", "")
            if tokenizer:
                total += len(tokenizer.encode(text, add_special_tokens=False))
            else:
                total += len(text) // 4
        return total

    # Keep old name for compat
    def count_recent_tokens(self) -> int:
        return self.count_tokens()

    def should_compress(self) -> bool:
        """Trigger when timeline tokens reach 80% of budget."""
        return self.count_tokens() >= COMPRESS_TOKEN_THRESHOLD

    def compress(self, summary: Dict, selected_indices: List[int]):
        """In-place replacement: selected timeline items → summary.

        selected_indices: positions in self.timeline to replace.
        Can include both thinks AND summaries (cross-summary compression).
        The new summary inherits source_chunks from all replaced items.
        """
        if not selected_indices:
            return

        # Collect source_chunks from all replaced items
        source_chunks = []
        for idx in sorted(selected_indices):
            item = self.timeline[idx]
            if item.get("type") == "think":
                source_chunks.append(item["chunk"])
            elif item.get("type") == "summary":
                source_chunks.extend(item.get("source_chunks", []))

        # Compute merge_level (max of replaced items + 1)
        max_level = 0
        for idx in selected_indices:
            item = self.timeline[idx]
            max_level = max(max_level, item.get("merge_level", 0))

        insert_pos = min(selected_indices)
        idx_set = set(selected_indices)

        # Remove selected items, insert new summary at first position
        new_timeline = []
        inserted = False
        for i, item in enumerate(self.timeline):
            if i in idx_set:
                if not inserted:
                    new_timeline.append({
                        "type": "summary",
                        "time_range": summary["time_range"],
                        "text": summary["text"],
                        "source_chunks": sorted(source_chunks),
                        "merge_level": max_level + 1,
                    })
                    inserted = True
                # skip replaced items
            else:
                new_timeline.append(item)

        self.timeline = new_timeline

    def format_for_prompt(self) -> str:
        """Format timeline as a single string for model input."""
        import json as _json
        lines = []
        for item in self.timeline:
            if item.get("type") == "summary":
                tr = item["time_range"]
                lines.append(f'<summary t="{tr[0]}-{tr[1]}">{item["text"]}</summary>')
            else:
                lines.append(f'[{item["time"]}] {item["text"]}')
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Observation Generation
# ---------------------------------------------------------------------------


def build_observation_request(
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
) -> Dict:
    """Build request for 397B to generate a student observation."""
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)

    memory_text = memory.format_for_prompt()

    prompt = OBSERVATION_PROMPT.format(
        compressed_memory="(see memory timeline below)",
        recent_thinks=memory_text or "(none)",
        window_start=int(window_start * AGENT_CHUNK_SEC),
        window_end=int(end),
        start=int(start),
        end=int(end),
    )

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
    """Parse observation output."""
    if raw is None:
        return "Scene continues without notable changes."
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = raw.strip('"').strip("'").strip()
    if len(raw) > 600:
        raw = raw[:600].rsplit(" ", 1)[0]
    return raw


# ---------------------------------------------------------------------------
# Compression: Range Selection (simplified v8.0)
# ---------------------------------------------------------------------------


def _evidence_by_chunk(evidence: Optional[List[Dict]]) -> Dict[int, Dict]:
    """Index teacher evidence by chunk_idx for O(1) lookup."""
    if not evidence:
        return {}
    return {cap.get("chunk_idx", i): cap for i, cap in enumerate(evidence)}


def score_range_for_compression(
    items: List[Dict],
    start_idx: int,
    timeline_len: int,
    evidence: Optional[List[Dict]] = None,
) -> float:
    """Score a candidate range for compression. LOWER = better to compress.

    Handles mixed ranges (thinks + summaries).

    Dimensions (normalized to comparable scales):
    - content_value [0~100]: high-value content → avoid compressing
    - merge_penalty [0~100]: re-compressing summaries → heavy penalty
    - boundary_penalty [0~30]: state_change at boundaries → penalize
    - recency_penalty [0~50]: recent items more valuable → penalize
    - token_saving [-50~0]: more tokens saved → lower score → prefer

    All dimensions contribute meaningfully; no single one dominates.
    """
    ev_index = _evidence_by_chunk(evidence) if evidence else {}

    # --- content_value: normalized per item ---
    content_value = 0.0
    for item in items:
        if item.get("type") == "think":
            cap = ev_index.get(item.get("chunk"))
            if cap:
                content_value += min(len(cap.get("visible_entities", [])) * 5.0, 20.0)
                if cap.get("ocr"):
                    content_value += 25.0  # OCR is high-value, hard to recall
                for fact in cap.get("atomic_facts", []):
                    if fact.get("confidence", 0) >= CONFIDENCE_THRESHOLD and any(c.isdigit() for c in fact.get("fact", "")):
                        content_value += 15.0  # numbers hard to recall
                content_value += len(cap.get("state_changes", [])) * 8.0
        elif item.get("type") == "summary":
            if any(c.isdigit() for c in item.get("text", "")):
                content_value += 10.0
    # Cap total content_value
    content_value = min(content_value, 100.0)

    # --- merge_penalty: re-compressing already compressed ---
    merge_penalty = 0.0
    for item in items:
        if item.get("type") == "summary":
            level = item.get("merge_level", 1)
            merge_penalty += 30.0 * level  # level 1→30, level 2→60, level 3→90

    # --- boundary_penalty ---
    boundary_penalty = 0.0
    if evidence:
        first_item = items[0]
        last_item = items[-1]
        if first_item.get("type") == "think":
            first_cap = ev_index.get(first_item.get("chunk"))
            if first_cap and first_cap.get("state_changes"):
                boundary_penalty += 15.0
        if last_item.get("type") == "think":
            last_cap = ev_index.get(last_item.get("chunk"))
            if last_cap and last_cap.get("state_changes"):
                boundary_penalty += 15.0

    # --- recency_penalty: recent items are more valuable ---
    # Position ratio: 0.0 (start of timeline) → 1.0 (end of timeline)
    if timeline_len > 0:
        center = (start_idx + len(items) / 2) / timeline_len
        recency_penalty = center * 50.0  # 0→0, 0.5→25, 1.0→50
    else:
        recency_penalty = 0.0

    # --- token_saving: normalize to [-50, 0] ---
    tokenizer = get_tokenizer()
    text = " ".join(item.get("text", "") for item in items)
    if tokenizer:
        n_tokens = sum(len(tokenizer.encode(item.get("text", ""), add_special_tokens=False)) for item in items)
    else:
        n_tokens = len(text) // 4
    # Normalize: 200 tokens → -50, 400 tokens → -50 (capped)
    token_saving = -min(n_tokens * 0.25, 50.0)

    return content_value + merge_penalty + boundary_penalty + recency_penalty + token_saving


def choose_optimal_compress_range(
    timeline: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> Tuple[List[int], Dict]:
    """Choose the best contiguous range in timeline to compress.

    Allows cross-summary ranges (thinks + summaries mixed).
    Summaries in the range get merge_level penalty in scoring.

    Returns: (selected_indices in timeline, policy_meta)
    """
    n = len(timeline)
    best_indices = None
    best_score = float("inf")

    # Enumerate all contiguous ranges of size 3 to COMPRESS_RANGE_MAX
    for size in range(COMPRESS_RANGE_MIN, min(COMPRESS_RANGE_MAX + 1, n + 1)):
        for start in range(0, n - size + 1):
            candidate_indices = list(range(start, start + size))
            candidate_items = [timeline[i] for i in candidate_indices]

            # Must contain at least 2 thinks (can't compress only summaries)
            n_thinks = sum(1 for it in candidate_items if it.get("type") == "think")
            if n_thinks < 2:
                continue

            score = score_range_for_compression(candidate_items, start, n, evidence)
            if score < best_score:
                best_score = score
                best_indices = candidate_indices

    if best_indices is None:
        # Fallback: last COMPRESS_RANGE_MIN items (most recent)
        think_indices = [i for i, t in enumerate(timeline) if t.get("type") == "think"]
        if len(think_indices) >= COMPRESS_RANGE_MIN:
            best_indices = think_indices[-COMPRESS_RANGE_MIN:]
        else:
            best_indices = think_indices

    meta = {
        "score": round(best_score, 2) if best_score < float("inf") else -1,
        "range_indices": best_indices,
        "range_size": len(best_indices) if best_indices else 0,
        "timeline_size": n,
        "n_thinks_in_range": sum(1 for i in (best_indices or []) if timeline[i].get("type") == "think"),
        "n_summaries_in_range": sum(1 for i in (best_indices or []) if timeline[i].get("type") == "summary"),
    }
    return best_indices, meta


# ---------------------------------------------------------------------------
# Compression: Summary Generation
# ---------------------------------------------------------------------------


def estimate_summary_length(
    observations: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> int:
    """Estimate appropriate summary length based on content complexity."""
    text = " ".join(item.get("text", item.get("obs", "")) for item in observations)
    has_numbers = any(c.isdigit() for c in text)

    n_entities = 0
    if evidence:
        ev_index = _evidence_by_chunk(evidence)
        for obs in observations:
            cap = ev_index.get(obs.get("chunk", -1))
            if cap:
                n_entities += len(cap.get("visible_entities", []))
        n_entities = min(n_entities, 15)
    else:
        words = set(w for w in text.split() if len(w) > 2)
        n_entities = len(words) // 5

    base = SUMMARY_TOKENS_MIN  # 100
    base += min(n_entities * 8, 60)
    base += 20 if has_numbers else 0

    return min(base, SUMMARY_TOKENS_MAX)


def build_compress_request(
    pre_action_timeline: List[Dict],
    memory: MemoryState,
    video_id: str,
    chunk_idx: int,
    evidence: Optional[List[Dict]] = None,
    frame_paths: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Build compression request from pre-action timeline.

    Finds contiguous think segments, selects best range within segments.
    """
    selected_indices, policy_meta = choose_optimal_compress_range(
        pre_action_timeline, evidence
    )

    if not selected_indices:
        return None

    to_compress = [pre_action_timeline[i] for i in selected_indices]

    obs_lines = []
    for item in to_compress:
        if item.get("type") == "think":
            obs_lines.append(f'[{item["time"]}] {item.get("text", "")}')
        elif item.get("type") == "summary":
            tr = item["time_range"]
            obs_lines.append(f'<summary t="{tr[0]}-{tr[1]}">{item.get("text", "")}</summary>')
    obs_text = "\n".join(obs_lines)

    # Compute time range from items (thinks have "chunk", summaries have "time_range")
    all_times = []
    for item in to_compress:
        if item.get("type") == "think":
            all_times.append(item["chunk"] * AGENT_CHUNK_SEC)
            all_times.append(item["chunk"] * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC)
        elif item.get("type") == "summary":
            all_times.extend(item["time_range"])
    first_time = min(all_times) if all_times else 0
    last_time = max(all_times) if all_times else 0

    target_length = estimate_summary_length(
        [item for item in to_compress if item.get("type") == "think"], evidence
    )

    # Overlapping frames: compress range ∩ visual window (only for thinks)
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    compress_chunks = [item["chunk"] for item in to_compress if item.get("type") == "think"]
    overlap_chunks = [c for c in compress_chunks if window_start <= c <= chunk_idx]

    overlap_frame_paths = []
    if frame_paths and overlap_chunks:
        for c in overlap_chunks:
            overlap_frame_paths.extend(get_chunk_frame_paths(frame_paths, c))

    visual_context = ""
    if overlap_frame_paths:
        overlap_start = overlap_chunks[0] * AGENT_CHUNK_SEC
        overlap_end = (overlap_chunks[-1] + 1) * AGENT_CHUNK_SEC
        visual_context = (
            f"\nVideo frames from t={int(overlap_start)}-{int(overlap_end)}s "
            f"are provided above for reference ({len(overlap_frame_paths)} frames). "
            f"Use them to verify entity details, counts, colors, and spatial positions.\n"
        )

    prompt = COMPRESS_PROMPT.format(
        observations_text=obs_text,
        visual_context=visual_context,
        target_length=target_length,
        start=int(first_time),
        end=int(last_time),
    )

    if overlap_frame_paths:
        content = build_vision_content(prompt, overlap_frame_paths)
    else:
        content = prompt

    return {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_compress"],
        "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
        "id": f"{video_id}_compress_{chunk_idx}",
        "_meta": {
            "time_range": [int(first_time), int(last_time)],
            "selected_indices": selected_indices,
            "chunks": compress_chunks,
            "teacher_policy": policy_meta,
            "overlap_chunks": overlap_chunks,
            "has_visual_context": bool(overlap_frame_paths),
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
            "time_range": meta["time_range"],
            "text": parsed.get("text", ""),
            "parse_success": True,
        }
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(raw)):
                if raw[i] == '{':
                    depth += 1
                elif raw[i] == '}':
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

    default["_raw"] = raw[:4000]
    return default


# ---------------------------------------------------------------------------
# Main Rollout
# ---------------------------------------------------------------------------


async def run_pass2_single_video(
    video_id: str,
    frame_paths: List[str],
    num_chunks: int,
    client,
    evidence: Optional[List[Dict]] = None,
) -> Dict:
    """Run question-blind streaming rollout for a single video.

    v8.0 changes:
    - No pending_questions (queries are independent, see §9.1)
    - Compression is conceptually BETWEEN timesteps (separate prompt)
    - Range scoring uses 3 dimensions (content_value + boundary_penalty + token_saving)
    - Logs compression range statistics for analysis
    """
    memory = MemoryState()
    thinks = []
    compression_events = []
    snapshots = {}

    for chunk_idx in range(num_chunks):
        # --- 1. Snapshot BEFORE this step's think ---
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)
        pre_action_timeline = snapshots[chunk_idx]["timeline"]
        pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]

        should_compress_now = (
            memory.should_compress()
            and len(pre_action_thinks) >= COMPRESS_RANGE_MIN
        )

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

        # --- 3. Compress (between timesteps) then append current think ---
        if should_compress_now:
            comp_request = build_compress_request(
                pre_action_timeline, memory, video_id, chunk_idx,
                evidence=evidence, frame_paths=frame_paths,
            )
            if comp_request is None:
                memory.add_think(chunk_idx, think_text)
                continue
            comp_raw = await client._call_one(
                messages=comp_request["messages"],
                max_tokens=comp_request["max_tokens"],
                temperature=comp_request["temperature"],
                request_id=comp_request["id"],
            )
            summary = parse_compress_result(comp_raw, comp_request["_meta"])
            selected_indices = comp_request["_meta"]["selected_indices"]

            memory.compress(summary, selected_indices=selected_indices)
            memory.add_think(chunk_idx, think_text)

            post_compress_tokens = memory.count_recent_tokens()
            hysteresis_ok = post_compress_tokens <= COMPRESS_HYSTERESIS_THRESHOLD
            if not hysteresis_ok:
                logger.warning(
                    f"  [{video_id}] Compression hysteresis violated at chunk {chunk_idx}: "
                    f"post-compress {post_compress_tokens} tok > {COMPRESS_HYSTERESIS_THRESHOLD} threshold"
                )

            compression_events.append({
                "trigger_chunk": chunk_idx,
                "summary": summary,
                "selected_indices": selected_indices,
                "compressed_thinks_chunks": comp_request["_meta"].get("chunks", []),
                "teacher_policy": comp_request["_meta"].get("teacher_policy", {}),
                "hysteresis_ok": hysteresis_ok,
                "post_compress_tokens": post_compress_tokens,
            })
            logger.debug(f"  [{video_id}] Compress at chunk {chunk_idx}: {summary['time_range']}")
        else:
            memory.add_think(chunk_idx, think_text)

        if (chunk_idx + 1) % 10 == 0:
            logger.info(
                f"  [{video_id}] Rollout: {chunk_idx+1}/{num_chunks} "
                f"(memory: {len(memory.recent_thinks)} thinks, "
                f"{len(memory.compressed_segments)} compressed)"
            )

    # --- Log compression statistics ---
    if compression_events:
        range_sizes = [len(e["compressed_thinks_chunks"]) for e in compression_events]
        range_durations = [
            (e["summary"]["time_range"][1] - e["summary"]["time_range"][0])
            for e in compression_events
        ]
        logger.info(
            f"  [{video_id}] Compression stats: {len(compression_events)} events, "
            f"range sizes: {range_sizes}, "
            f"durations(s): {range_durations}"
        )

    return {
        "video_id": video_id,
        "num_chunks": num_chunks,
        "thinks": thinks,
        "compression_events": compression_events,
        "snapshots": snapshots,
        "final_memory": memory.snapshot(num_chunks),
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_rollout(video_id: str, rollout: Dict, output_dir: Path = ROLLOUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rollout, f, ensure_ascii=False)


def load_rollout(video_id: str, rollout_dir: Path = ROLLOUT_DIR) -> Optional[Dict]:
    path = rollout_dir / f"{video_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "snapshots" in data:
        data["snapshots"] = {int(k): v for k, v in data["snapshots"].items()}
    return data


# ---------------------------------------------------------------------------
# Compression Statistics (run after all videos complete)
# ---------------------------------------------------------------------------


def compute_compression_stats(rollout_map: Dict[str, Dict]) -> Dict:
    """Aggregate compression statistics across all videos.

    Run after Pass 2 completes. Outputs report for tuning
    COMPRESS_RANGE_MIN/MAX and diagnosing quality issues.

    Args:
        rollout_map: {video_id: rollout_dict}

    Returns:
        Stats dict (also logged). Save to audit dir.
    """
    all_range_sizes = []
    all_durations = []
    all_post_tokens = []
    hysteresis_violations = 0
    total_events = 0
    parse_success = 0
    parse_fail = 0
    videos_with_no_compression = 0

    for vid, rollout in rollout_map.items():
        events = rollout.get("compression_events", [])
        if not events:
            videos_with_no_compression += 1
            continue
        for event in events:
            total_events += 1
            chunks = event.get("compressed_thinks_chunks", [])
            all_range_sizes.append(len(chunks))
            tr = event.get("summary", {}).get("time_range", [0, 0])
            all_durations.append(tr[1] - tr[0])
            all_post_tokens.append(event.get("post_compress_tokens", 0))
            if not event.get("hysteresis_ok", True):
                hysteresis_violations += 1
            if event.get("summary", {}).get("parse_success", False):
                parse_success += 1
            else:
                parse_fail += 1

    def _percentiles(values, pcts=(25, 50, 75, 90)):
        if not values:
            return {}
        s = sorted(values)
        return {f"p{p}": s[min(len(s) - 1, int(len(s) * p / 100))] for p in pcts}

    stats = {
        "total_videos": len(rollout_map),
        "videos_with_compression": len(rollout_map) - videos_with_no_compression,
        "videos_without_compression": videos_with_no_compression,
        "total_compression_events": total_events,
        "avg_events_per_video": round(total_events / max(len(rollout_map), 1), 1),

        "range_size": {
            "min": min(all_range_sizes) if all_range_sizes else 0,
            "max": max(all_range_sizes) if all_range_sizes else 0,
            "mean": round(sum(all_range_sizes) / max(len(all_range_sizes), 1), 1),
            **_percentiles(all_range_sizes),
            "distribution": {
                s: all_range_sizes.count(s) for s in range(
                    COMPRESS_RANGE_MIN, COMPRESS_RANGE_MAX + 1
                )
            },
        },

        "duration_sec": {
            "min": min(all_durations) if all_durations else 0,
            "max": max(all_durations) if all_durations else 0,
            "mean": round(sum(all_durations) / max(len(all_durations), 1), 1),
            **_percentiles(all_durations),
        },

        "hysteresis": {
            "violations": hysteresis_violations,
            "total": total_events,
            "violation_rate": round(hysteresis_violations / max(total_events, 1), 3),
            "post_compress_tokens": _percentiles(all_post_tokens),
        },

        "summary_parse": {
            "success": parse_success,
            "fail": parse_fail,
            "success_rate": round(parse_success / max(parse_success + parse_fail, 1), 3),
        },
    }

    # Log summary
    logger.info("=" * 50)
    logger.info("COMPRESSION STATISTICS")
    logger.info(f"  Events: {total_events} across {stats['videos_with_compression']} videos")
    logger.info(f"  Range size: mean={stats['range_size']['mean']}, "
                f"distribution={stats['range_size']['distribution']}")
    logger.info(f"  Duration: mean={stats['duration_sec']['mean']}s, "
                f"p50={stats['duration_sec'].get('p50', '?')}s, "
                f"p90={stats['duration_sec'].get('p90', '?')}s")
    logger.info(f"  Hysteresis violations: {hysteresis_violations}/{total_events} "
                f"({stats['hysteresis']['violation_rate']:.1%})")
    logger.info(f"  Summary parse: {parse_success}/{parse_success+parse_fail} "
                f"({stats['summary_parse']['success_rate']:.1%})")
    logger.info("=" * 50)

    return stats
