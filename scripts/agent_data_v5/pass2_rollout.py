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

    def compress(self, summary: Dict, compressed_chunks: List[int]):
        """In-place replacement: selected thinks → summary at same position.

        Finds the first think in compressed_chunks, inserts summary there,
        then removes all thinks in compressed_chunks. Timeline stays sorted.
        """
        chunk_set = set(compressed_chunks)

        # Find insertion position (where first compressed think is)
        insert_pos = None
        for i, item in enumerate(self.timeline):
            if item.get("type") == "think" and item.get("chunk") in chunk_set:
                insert_pos = i
                break

        # Remove compressed thinks
        self.timeline = [t for t in self.timeline if not (
            t.get("type") == "think" and t.get("chunk") in chunk_set
        )]

        # Insert summary at the position where first think was
        summary_item = {
            "type": "summary",
            "time_range": summary["time_range"],
            "text": summary["text"],
        }
        if insert_pos is not None:
            # Adjust insert_pos for removed items before it
            adjusted = sum(1 for t in self.timeline[:insert_pos]
                           if t.get("type") != "_removed")
            self.timeline.insert(min(adjusted, len(self.timeline)), summary_item)
        else:
            self.timeline.append(summary_item)

        # Merge oldest two summaries if too many
        summaries = [i for i, t in enumerate(self.timeline) if t.get("type") == "summary"]
        if len(summaries) > MAX_COMPRESSED_SEGMENTS:
            idx_a, idx_b = summaries[0], summaries[1]
            seg_a = self.timeline[idx_a]
            seg_b = self.timeline[idx_b]
            combined = f'{seg_a["text"]} {seg_b["text"]}'
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
                "type": "summary",
                "time_range": [seg_a["time_range"][0], seg_b["time_range"][1]],
                "text": combined,
                "merged": True,
            }
            # Remove both, insert merged at first position
            self.timeline = [t for i, t in enumerate(self.timeline)
                             if i not in (idx_a, idx_b)]
            self.timeline.insert(idx_a, merged)

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
    thinks: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> float:
    """Score a candidate range for compression. LOWER = better to compress.

    v8.0 simplified to 3 dimensions (removed pending_overlap, reconstructability):
    - content_value: high-value content (entities, numbers, OCR) → avoid compressing
    - boundary_penalty: range starts/ends at state_change → splitting event → penalize
    - token_saving: more tokens saved → lower score → prefer
    """
    text = " ".join(item.get("text", item.get("obs", "")) for item in thinks)

    # --- content_value: from teacher evidence ---
    content_value = 0.0
    if evidence:
        ev_index = _evidence_by_chunk(evidence)
        for t in thinks:
            cap = ev_index.get(t["chunk"])
            if not cap:
                continue
            # Entity count
            content_value += len(cap.get("visible_entities", [])) * 2.0
            # OCR
            if cap.get("ocr"):
                content_value += 8.0
            # Numbers in facts
            for fact in cap.get("atomic_facts", []):
                if fact.get("confidence", 0) >= CONFIDENCE_THRESHOLD:
                    if any(c.isdigit() for c in fact.get("fact", "")):
                        content_value += 4.0
            # State changes (from 1-B)
            content_value += len(cap.get("state_changes", [])) * 3.0

    # --- boundary_penalty: from 1-B state_changes ---
    boundary_penalty = 0.0
    if evidence:
        ev_index = _evidence_by_chunk(evidence)
        # Penalize if range STARTS at a state change (splitting from prev event)
        first_cap = ev_index.get(thinks[0]["chunk"])
        if first_cap and first_cap.get("state_changes"):
            boundary_penalty += 5.0
        # Penalize if range ENDS at a state change (splitting ongoing event)
        last_cap = ev_index.get(thinks[-1]["chunk"])
        if last_cap and last_cap.get("state_changes"):
            boundary_penalty += 5.0

    # --- token_saving: precise token count ---
    tokenizer = get_tokenizer()
    if tokenizer:
        n_tokens = sum(
            len(tokenizer.encode(item.get("text", ""), add_special_tokens=False))
            for item in thinks
        )
    else:
        n_tokens = len(text) // 4
    token_saving = -n_tokens * 0.3

    return content_value + boundary_penalty + token_saving


def _find_contiguous_think_segments(timeline: List[Dict]) -> List[List[Dict]]:
    """Find contiguous think segments in the timeline (not separated by summaries).

    Example:
      timeline: [think_0, think_1, summary, think_6, think_7, think_8, think_9]
      returns: [[think_0, think_1], [think_6, think_7, think_8, think_9]]

    Only segments with >= COMPRESS_RANGE_MIN thinks are compression candidates.
    """
    segments = []
    current = []
    for item in timeline:
        if item.get("type") == "think":
            current.append(item)
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        segments.append(current)
    return segments


def choose_optimal_compress_range(
    timeline: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], Dict]:
    """Choose the best contiguous think range to compress.

    Finds contiguous think segments in the timeline (no summary in between),
    then enumerates valid ranges within each segment.

    Returns: (best_range, policy_meta)
    """
    segments = _find_contiguous_think_segments(timeline)
    best_range = None
    best_score = float("inf")
    best_segment_idx = 0
    best_start = 0
    best_size = 0

    for seg_idx, segment in enumerate(segments):
        n = len(segment)
        if n < COMPRESS_RANGE_MIN:
            continue  # Segment too small
        for size in range(COMPRESS_RANGE_MIN, min(COMPRESS_RANGE_MAX + 1, n + 1)):
            for start in range(0, n - size + 1):
                candidate = segment[start:start + size]
                score = score_range_for_compression(candidate, evidence)
                if score < best_score:
                    best_score = score
                    best_range = candidate
                    best_segment_idx = seg_idx
                    best_start = start
                    best_size = size

    if best_range is None:
        # Fallback: largest segment, first COMPRESS_RANGE_MIN items
        largest = max(segments, key=len) if segments else []
        if len(largest) >= COMPRESS_RANGE_MIN:
            best_range = largest[:COMPRESS_RANGE_MIN]
        else:
            # No valid segment at all
            all_thinks = [t for t in timeline if t.get("type") == "think"]
            best_range = all_thinks[:COMPRESS_RANGE_MIN] if len(all_thinks) >= COMPRESS_RANGE_MIN else all_thinks

    meta = {
        "score": round(best_score, 2) if best_score < float("inf") else -1,
        "segment_idx": best_segment_idx,
        "range_start_in_segment": best_start,
        "range_size": best_size or len(best_range),
        "n_segments": len(segments),
        "segment_sizes": [len(s) for s in segments],
    }
    return best_range, meta


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
    to_compress, policy_meta = choose_optimal_compress_range(
        pre_action_timeline, evidence
    )

    if not to_compress:
        return None

    obs_lines = [f'[{item["time"]}] {item.get("text", "")}' for item in to_compress]
    obs_text = "\n".join(obs_lines)

    first_time = to_compress[0]["chunk"] * AGENT_CHUNK_SEC
    last_time = to_compress[-1]["chunk"] * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC

    target_length = estimate_summary_length(to_compress, evidence)

    # Overlapping frames: compress range ∩ visual window
    window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
    compress_chunks = [item["chunk"] for item in to_compress]
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
            real_chunks = comp_request["_meta"]["chunks"]

            memory.compress(summary, compressed_chunks=real_chunks)
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
                "compressed_thinks_chunks": real_chunks,
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
