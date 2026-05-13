"""
Pass 2: Question-blind Streaming Rollout

Simulates the student model's real streaming experience WITHOUT any questions.
Generates memory snapshots and compression decisions.

v12.25: current-chunk think text comes from pass1's independent ``think``
observation-note JSON field (not Qwen/vLLM enable_thinking reasoning;
fallback: deterministic rendering from pass1 evidence). Pass2 no longer asks
the teacher to generate observation notes from full text memory plus the
sliding visual window, because that setup repeatedly copied stale history into
current thinks. The only remaining teacher call in pass2 is the text-only
compression summary request.

Key principle: Question-blind — no future question knowledge influences this pass.
Compression summaries use ONLY student observations (not teacher captions).
Compression is triggered BETWEEN timesteps (separate prompt, does not occupy a timestep).

Processing: Sequential per video, parallel across videos.
"""

import json
import logging
import re
import html
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

from .config import (
    AGENT_CHUNK_SEC,
    COMPACT_MEMORY_BALANCE_SEGMENTS,
    COMPACT_MEMORY_MAX_NEW_CHUNKS,
    COMPACT_MEMORY_MIN_NEW_CHUNKS,
    COMPACT_MEMORY_TARGET_NEW_CHUNKS,
    COMPACT_MEMORY_TEXT_TOKEN_BUDGET,
    COMPACT_MEMORY_UPDATE_MODE,
    COMPACT_MEMORY_UPDATE_PROMPT,
    COMPACT_MEMORY_UPDATE_SYSTEM_PROMPT,
    COMPRESS_PROMPT,
    COMPRESS_RANGE_MAX,
    COMPRESS_RANGE_MIN,
    COMPRESS_TOKEN_THRESHOLD,
    COMPRESS_HYSTERESIS_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    FRAMES_PER_CHUNK,
    OBSERVATION_PROMPT,
    OBSERVATION_REPAIR_PROMPT,
    get_tokenizer,
    PASS_CONFIG,
    RUNTIME_MM_PROCESSOR_KWARGS,
    ROLLOUT_DIR,
    MAX_COMPRESSED_SEGMENTS,
    SUMMARY_TOKENS_MAX,
    SUMMARY_TOKENS_MIN,
    VISUAL_WINDOW_CHUNKS,
    compute_visual_window_start,
)
from .evidence_think import build_think_from_pass1_evidence, think_source_for_evidence
from .pass1a_evidence import get_chunk_frame_paths
from scripts.agent_data_pipeline.vllm_client import TruncatedCompletionError, encode_image_base64
from thinkstream.data.agent_protocol import append_timestamped_image_list

logger = logging.getLogger(__name__)


def plan_compact_memory_intervals(
    num_chunks: int,
    *,
    min_len: int = COMPACT_MEMORY_MIN_NEW_CHUNKS,
    target_len: int = COMPACT_MEMORY_TARGET_NEW_CHUNKS,
    max_len: int = COMPACT_MEMORY_MAX_NEW_CHUNKS,
) -> List[int]:
    """Plan full-video compact-memory segment lengths.

    The online runtime can only use text-token pressure, but pass2 data
    construction knows the full video length. Use that to avoid pathological
    short tails from fixed max-length triggering, e.g. 149 -> 36/36/36/36/5.
    If an exact 25-36 partition is mathematically impossible, prefer one
    slightly-longer segment over a very short tail.
    """
    n = int(num_chunks or 0)
    if n <= 0:
        return []
    min_len = max(1, int(min_len))
    target_len = max(min_len, int(target_len))
    max_len = max(min_len, int(max_len))
    if n <= max_len:
        return [n]

    min_segments = (n + max_len - 1) // max_len
    max_segments = n // min_len
    if min_segments <= max_segments:
        target_segments = max(1, round(n / target_len))
        k = min(max(target_segments, min_segments), max_segments)
        base, rem = divmod(n, k)
        return [base + (1 if i < rem else 0) for i in range(k)]

    # Infeasible lengths are mostly 37-49 and 73-74 for the default 25-36
    # range. Choose the largest segment count that still keeps every segment
    # at least min_len; this creates a slightly-long segment instead of a
    # short tail, e.g. 43 -> 43 and 73 -> 37/36.
    k = max(1, max_segments)
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def plan_compact_memory_boundaries(
    num_chunks: int,
    *,
    min_len: int = COMPACT_MEMORY_MIN_NEW_CHUNKS,
    target_len: int = COMPACT_MEMORY_TARGET_NEW_CHUNKS,
    max_len: int = COMPACT_MEMORY_MAX_NEW_CHUNKS,
) -> List[int]:
    """Return chunk indices where pass2 should compact before that chunk."""
    pos = 0
    boundaries: List[int] = []
    intervals = plan_compact_memory_intervals(
        num_chunks,
        min_len=min_len,
        target_len=target_len,
        max_len=max_len,
    )
    for length in intervals[:-1]:
        pos += int(length)
        boundaries.append(pos)
    return boundaries


# v12.11 hotfix (2026-05-01): vLLM rejects requests where the server-side
# `max_model_len - prompt_tokens` < requested max_tokens with the misleading
# error "max_tokens must be at least 1, got -<N>". For pass2's long-memory
# rollouts the prompt grows ~14 tok/chunk; without client-side capping a
# 100-chunk video accumulates >2K tokens of memory text past the server's
# usable budget.
#
# User intent (2026-05-01): pass2 runs with max_tokens=16K and
# concurrent_videos=1024 against a vLLM with VLLM_MAX_MODEL_LEN=65536
# (config.py:279). Under that deployment the cap is 65536 - input - margin,
# which only kicks in when input > ~49K (rare; long videos with full memory
# can approach this). For deployments with smaller max_model_len, override
# via THINKSTREAM_VLLM_MAX_MODEL_LEN env so this client-side cap matches
# the server's actual context window.
#
# Conservative input estimate per pass2 observation request (v12.15):
#   visual:  16 frames × ~235 tok/frame = ~3,760 tok (RUNTIME mm_processor_kwargs)
#   memory:  recent_thinks ≤ 4000 tok + compressed ≤ 1400 tok = ~5400 tok
#   prompt template + safety: ~700 tok
#   ─────────────────────────────────────────────────
#   estimated input ~9,800 tok worst case → max_tokens=16K fits in 64K cap.
import os as _os
_PASS2_SAFE_MAX_MODEL_LEN = int(
    _os.environ.get("THINKSTREAM_VLLM_MAX_MODEL_LEN", "65536")
)
_PASS2_INPUT_MARGIN = 1500           # tokenizer drift + safety margin

# v12.12: vision token estimate must match RUNTIME_MM_PROCESSOR_KWARGS profile.
# Empirically ~235 tok/frame at min=130k max=220k (config.py).
from .config import VISUAL_TOKENS_PER_FRAME_RUNTIME as _PASS2_VISION_TOKENS_PER_FRAME


def _safe_max_tokens_for_pass2(
    request: Dict,
    configured_max: int,
    *,
    floor: int = 1024,  # v12.11: matches observation min target (think
                        # output 40-80 tok with 12× safety). Compress
                        # callers pass configured_max=4096 which also
                        # stays ≥ floor; JSON-output truncation prevention.
) -> int:
    """Compute a max_tokens value that won't trip the vLLM context cap.

    Estimates input tokens from message content (text via len/3, vision via
    count × VISUAL_TOKENS_PER_FRAME_RUNTIME). Returns max(floor, min(configured, max_model_len - input - margin)).
    """
    msgs = request.get("messages", [])
    media_video = (request.get("media_io_kwargs") or {}).get("video") or {}
    n_text_chars = 0
    n_video_frames = 0
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            n_text_chars += len(c)
        elif isinstance(c, list):
            for it in c:
                if not isinstance(it, dict):
                    continue
                if it.get("type") in ("text",):
                    n_text_chars += len(it.get("text", ""))
                elif it.get("type") in ("video", "video_url", "image_url", "image"):
                    # Vision item: each frame ≈ VISUAL_TOKENS_PER_FRAME_RUNTIME
                    # tok (RUNTIME mm_processor_kwargs profile).
                    if it.get("type") == "video":
                        v = it.get("video")
                        if isinstance(v, list):
                            n_video_frames += len(v)
                        elif isinstance(v, str):
                            n_video_frames += 1
                    elif it.get("type") == "video_url":
                        # Legacy/raw-video fallback; normal pass2 requests
                        # now use one image_url item per timestamped frame.
                        idx = media_video.get("frames_indices")
                        n_video_frames += len(idx) if isinstance(idx, list) else 1
                    else:
                        n_video_frames += 1
    estimated_input = (
        n_text_chars // 3
        + n_video_frames * _PASS2_VISION_TOKENS_PER_FRAME
        + _PASS2_INPUT_MARGIN
    )
    available = _PASS2_SAFE_MAX_MODEL_LEN - estimated_input
    safe = max(floor, min(int(configured_max), int(available)))
    return safe


# ---------------------------------------------------------------------------
# Memory State
# ---------------------------------------------------------------------------


class MemoryState:
    """Tracks the student model's text memory at each timestep.

    Timeline stores visible memory in chronological order: raw think items and
    compressed summaries share one list. ``compressed_segments`` and
    ``recent_thinks`` are derived views used for compatibility with downstream
    renderers and runtime state.

    Queries managed in separate active-query/response-history zones, independent of memory.
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
            "visual_window_start": compute_visual_window_start(chunk_idx),
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

    def _count_item_tokens(self, item: Dict) -> int:
        text = item.get("text", "")
        tokenizer = get_tokenizer()
        if tokenizer:
            return len(tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def count_tokens(self) -> int:
        """Count total visible-memory tokens (thinks + summaries)."""
        total = 0
        for item in self.timeline:
            total += self._count_item_tokens(item)
        return total

    def count_recent_tokens(self) -> int:
        """Backward-compatible alias for full visible-memory token count."""
        return self.count_tokens()

    def compress_trigger_diagnostic(self) -> Dict:
        """Return compact-memory trigger status and the accounting inputs.

        The production trigger intentionally ignores visual tokens and tool
        output tokens; those are reserved at the context-budget level. This
        counter tracks only visible text memory: previous compact summaries
        plus raw observations since the last compact update.
        """
        if not COMPACT_MEMORY_UPDATE_MODE:
            tokens = self.count_recent_tokens()
            return {
                "triggered": tokens >= COMPRESS_TOKEN_THRESHOLD,
                "mode": "legacy_range_summary",
                "visible_text_tokens": tokens,
                "threshold_tokens": COMPRESS_TOKEN_THRESHOLD,
                "recent_raw_chunks": len(self.recent_thinks),
                "min_new_chunks": COMPRESS_RANGE_MIN,
                "max_new_chunks": COMPRESS_RANGE_MAX,
                "reason": "legacy_token_threshold" if tokens >= COMPRESS_TOKEN_THRESHOLD else "below_threshold",
            }

        tokens = self.count_tokens()
        recent_raw = len(self.recent_thinks)
        token_ready = tokens >= COMPACT_MEMORY_TEXT_TOKEN_BUDGET
        forced_by_age = recent_raw >= COMPACT_MEMORY_MAX_NEW_CHUNKS
        enough_new = recent_raw >= COMPACT_MEMORY_MIN_NEW_CHUNKS
        triggered = enough_new and (token_ready or forced_by_age)
        if triggered:
            reason = "text_budget" if token_ready else "max_new_chunks"
        elif not enough_new:
            reason = "min_new_chunks"
        else:
            reason = "below_text_budget"
        return {
            "triggered": triggered,
            "mode": "compact_memory_update",
            "visible_text_tokens": tokens,
            "threshold_tokens": COMPACT_MEMORY_TEXT_TOKEN_BUDGET,
            "recent_raw_chunks": recent_raw,
            "min_new_chunks": COMPACT_MEMORY_MIN_NEW_CHUNKS,
            "target_new_chunks": COMPACT_MEMORY_TARGET_NEW_CHUNKS,
            "max_new_chunks": COMPACT_MEMORY_MAX_NEW_CHUNKS,
            "token_ready": token_ready,
            "forced_by_age": forced_by_age,
            "reason": reason,
        }

    def should_compress(self) -> bool:
        """Trigger when the full visible text memory reaches the budget."""
        return bool(self.compress_trigger_diagnostic().get("triggered"))

    def compress(self, summary: Dict, selected_indices: List[int]):
        """In-place replacement: selected timeline items → summary.

        selected_indices: positions in self.timeline to replace.
        Can include both thinks AND summaries (cross-summary compression).
        The new summary inherits source_chunks from all replaced items.
        """
        if not selected_indices:
            return

        # Collect source chunks from all replaced items. Older caches did not
        # persist source_chunks on summaries, so fall back to their time_range.
        source_chunks = []
        for idx in sorted(selected_indices):
            source_chunks.extend(_item_source_chunks(self.timeline[idx]))

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
                        "source_chunks": sorted(set(source_chunks)),
                        "merge_level": max_level + 1,
                    })
                    inserted = True
                # skip replaced items
            else:
                new_timeline.append(item)

        self.timeline = new_timeline

    def replace_with_compact_entries(self, summary: Dict):
        """Replace the visible timeline with teacher-produced compact memory.

        Retrieval archive is deliberately untouched so recall can still map
        raw one-second observations back to original frames.
        """
        entries = summary.get("entries") or []
        new_timeline: List[Dict] = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            tr = entry.get("time_range") or []
            if not text or not (isinstance(tr, list) and len(tr) == 2):
                continue
            try:
                start, end = int(tr[0]), int(tr[1])
            except (TypeError, ValueError):
                continue
            if end < start:
                start, end = end, start
            source_chunks: List[int] = []
            for c in (entry.get("source_chunks") or range(start, end + 1)):
                try:
                    source_chunks.append(int(c))
                except (TypeError, ValueError):
                    continue
            new_timeline.append({
                "type": "summary",
                "time_range": [start, end],
                "text": text,
                "source_chunks": sorted(set(source_chunks)),
                "merge_level": int(entry.get("merge_level", summary.get("merge_level", 1)) or 1),
                "compact_memory": True,
            })
        self.timeline = sorted(new_timeline, key=lambda item: (item["time_range"][0], item["time_range"][1]))

    @staticmethod
    def _format_timeline_item_as_memory_tag(item: Dict) -> str:
        """Render one timeline item with the same tags the student sees."""
        if item.get("type") == "summary":
            payload = {
                "time_range": list(item.get("time_range") or []),
                "text": item.get("text", ""),
            }
            return f"<compressed>{json.dumps(payload, ensure_ascii=False)}</compressed>"
        payload = {
            "time": item.get("time", ""),
            "text": item.get("text", ""),
        }
        return f"<memory_think>{json.dumps(payload, ensure_ascii=False)}</memory_think>"

    def format_for_prompt(self) -> str:
        """Format timeline with the same memory tags used by SFT/RL/eval."""
        return "\n".join(
            self._format_timeline_item_as_memory_tag(item)
            for item in self.timeline
        )

    def format_for_observation_prompt(self) -> str:
        """Serialize memory for observation as tagged archival records.

        Keep the full compression + full memory contents, but render them as
        the same machine-readable tags used by the student prompt. This
        preserves information while making it harder for the model to continue
        prior narration verbatim when the latest frames differ.
        """
        return self.format_for_prompt()

    def format_recent_for_repair_prompt(self, limit: int = 8) -> str:
        """Serialize recent thinks for repair as tagged archival records."""
        records = [
            self._format_timeline_item_as_memory_tag(item)
            for item in self.recent_thinks[-max(0, int(limit)):]
        ]
        return "\n".join(records) or "(none)"


# ---------------------------------------------------------------------------
# Observation Generation
# ---------------------------------------------------------------------------

_META_REASONING_RE = re.compile(
    r"^\s*(?:"
    r"the user wants|the task|we need|i need|i should|i will|let'?s|"
    r"analy[sz]e the frames|looking at the frames|from the frames"
    r")\b",
    re.IGNORECASE,
)
_PLACEHOLDER_TEXT_RE = re.compile(r"^[.\s…-]+$")
_STALE_OBSERVATION_RE = re.compile(
    r"\b(continues?|remains?|unchanged|no new|static|identical)\b",
    re.IGNORECASE,
)
_FRAME_TAG_LINE_RE = re.compile(
    r'^\s*<frame\s+ts="[^"]+"\s+role="[^"]+"\s*/>\s*$',
    re.IGNORECASE,
)
_FRAME_TAG_INLINE_RE = re.compile(
    r'\s*<frame\s+ts="[^"]+"\s+role="[^"]+"\s*/>\s*',
    re.IGNORECASE,
)
_REPAIR_ACCEPT_SIMILARITY_MAX = 0.82
_STATIC_REPAIR_MSE_MAX = 50.0


def _norm_observation(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _observation_tokens(text: str) -> set:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "over",
        "under", "left", "right", "center", "middle", "frame", "video",
        "scene", "visible", "text", "still", "same", "latest", "second",
        "continues", "continue", "remain", "remains", "unchanged", "static",
        "during", "throughout", "current",
    }
    return {
        tok for tok in _norm_observation(text).split()
        if len(tok) > 2 and tok not in stop
    }


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def _evidence_text(chunk: Dict) -> str:
    parts: List[str] = []
    for ent in chunk.get("visible_entities") or []:
        if isinstance(ent, dict):
            parts.append(str(ent.get("desc", "")))
            parts.append(str(ent.get("action", "")))
    for fact in chunk.get("atomic_facts") or []:
        if isinstance(fact, dict):
            parts.append(str(fact.get("fact", "")))
        else:
            parts.append(str(fact))
    for change in chunk.get("state_changes") or []:
        parts.append(str(change))
    return " ".join(p for p in parts if p)


def _evidence_drift(evidence: Optional[List[Dict]], start: int, end: int) -> Optional[float]:
    if not evidence or start < 0 or end < 0:
        return None
    if start >= len(evidence) or end >= len(evidence):
        return None
    return 1.0 - _jaccard(
        _observation_tokens(_evidence_text(evidence[start])),
        _observation_tokens(_evidence_text(evidence[end])),
    )


def _frame_pair_mse(prev_path: str, cur_path: str) -> Optional[float]:
    try:
        with Image.open(prev_path).convert("RGB") as prev_img:
            prev = list(prev_img.getdata())
        with Image.open(cur_path).convert("RGB") as cur_img:
            cur = list(cur_img.getdata())
    except Exception:
        return None

    if len(prev) != len(cur) or not prev:
        return None

    total = 0.0
    for (pr, pg, pb), (cr, cg, cb) in zip(prev, cur):
        total += (pr - cr) ** 2 + (pg - cg) ** 2 + (pb - cb) ** 2
    return total / (len(prev) * 3.0)


def _chunk_visual_delta_mse(
    frame_paths: List[str],
    chunk_idx: int,
) -> Optional[float]:
    if chunk_idx <= 0:
        return None
    prev_chunk = get_chunk_frame_paths(frame_paths, chunk_idx - 1)
    cur_chunk = get_chunk_frame_paths(frame_paths, chunk_idx)
    vals: List[float] = []
    for prev_path, cur_path in zip(prev_chunk, cur_chunk):
        if not Path(prev_path).exists() or not Path(cur_path).exists():
            continue
        mse = _frame_pair_mse(prev_path, cur_path)
        if mse is not None:
            vals.append(mse)
    if not vals:
        return None
    return sum(vals) / len(vals)


def should_repair_observation(
    think_text: str,
    recent_thinks: List[Dict],
    *,
    chunk_idx: int,
    evidence: Optional[List[Dict]] = None,
    visual_delta_mse: Optional[float] = None,
) -> Tuple[bool, Dict]:
    """Detect stale-copy pass2 observations before they enter memory.

    Pass2's gold think is reused directly by SFT/RL samples. A repeated
    "continues/remains unchanged" sentence is acceptable for truly static
    frames, but it is harmful when the visual evidence has drifted. This gate
    uses pass1 evidence only to decide whether to retry; the repair prompt
    itself still receives only video frames, so no pass1 text leaks into the
    target think.
    """
    norm = _norm_observation(think_text)
    if not norm or not recent_thinks:
        return False, {}

    prev_texts = [str(t.get("text", "")) for t in recent_thinks if t.get("type") == "think"]
    exact_prev = 0
    for prev in reversed(prev_texts):
        if _norm_observation(prev) == norm:
            exact_prev += 1
        else:
            break

    cur_tokens = _observation_tokens(think_text)
    near_prev = 0
    for prev in reversed(prev_texts):
        if _jaccard(_observation_tokens(prev), cur_tokens) >= 0.86:
            near_prev += 1
        else:
            break

    exact_len = exact_prev + 1
    near_len = near_prev + 1
    candidates: List[Tuple[str, int, int]] = []
    if exact_len >= 4:
        candidates.append(("exact", exact_len, chunk_idx - exact_prev))
    if near_len >= 6:
        candidates.append(("near", near_len, chunk_idx - near_prev))
    if not candidates:
        return False, {}

    if visual_delta_mse is not None and visual_delta_mse < _STATIC_REPAIR_MSE_MAX:
        return False, {
            "reason": "static_visual_delta_skip",
            "visual_delta_mse": round(visual_delta_mse, 3),
        }

    best_kind, best_len, best_start = max(candidates, key=lambda x: x[1])
    stale_language = bool(_STALE_OBSERVATION_RE.search(think_text))
    drift = _evidence_drift(evidence, best_start, chunk_idx)
    if drift is not None:
        # Old gate required stale words such as "continues/remains". The
        # production failure repeated high-similarity factual sentences without
        # those words, so evidence drift now carries near/exact-repeat repair.
        threshold = 0.45 if stale_language else 0.65
        if drift < threshold:
            return False, {}
        return True, {
            "reason": f"{best_kind}_repeat_with_evidence_drift",
            "run_length": best_len,
            "run_start": best_start,
            "evidence_drift": round(drift, 3),
        }

    # If pass1b evidence is unavailable for this video, keep the fallback
    # conservative to avoid doubling calls on genuinely static title screens.
    if (
        (stale_language and best_kind == "exact" and best_len >= 12)
        or (stale_language and best_len >= 16)
        or (best_kind == "exact" and best_len >= 18)
        or best_len >= 24
    ):
        return True, {
            "reason": f"{best_kind}_repeat_without_evidence",
            "run_length": best_len,
            "run_start": best_start,
            "evidence_drift": None,
        }
    return False, {}


def _is_repair_better(
    repaired_text: str,
    stale_text: str,
    recent_thinks: List[Dict],
) -> bool:
    """Accept a repair only if it breaks away from stale recent memory."""
    if not repaired_text or repaired_text == "Scene continues without notable changes.":
        return False
    repaired_tokens = _observation_tokens(repaired_text)
    if not repaired_tokens:
        return False
    if _jaccard(repaired_tokens, _observation_tokens(stale_text)) >= _REPAIR_ACCEPT_SIMILARITY_MAX:
        return False
    for prev in recent_thinks[-3:]:
        if _jaccard(
            repaired_tokens,
            _observation_tokens(str(prev.get("text", ""))),
        ) >= _REPAIR_ACCEPT_SIMILARITY_MAX:
            return False
    return True


def build_observation_request(
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
) -> Dict:
    """Build request for 397B to generate a student observation.

    Fallback-only path. Production pass2 uses pass1's current-only evidence
    notes; if evidence is missing, keep this teacher request aligned with the
    runtime prompt by sending only the current 1s chunk's two frames. The
    recurrent visual KV window is a runtime mechanism and is not replayed in
    the request content.
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    window_start = chunk_idx

    memory_text = memory.format_for_observation_prompt()

    prompt = OBSERVATION_PROMPT.format(
        compressed_memory="(see memory timeline below)",
        recent_thinks=memory_text or "(none)",
        window_start=int(window_start * AGENT_CHUNK_SEC),
        window_end=int(end),
        start=int(start),
        end=int(end),
        current_frame_count=FRAMES_PER_CHUNK,
    )

    content: List[Dict] = [{"type": "text", "text": prompt}]
    window_images: List[str] = []
    timestamp_labels: List[str] = []
    for img_path in get_chunk_frame_paths(frame_paths, chunk_idx):
        if not Path(img_path).exists():
            continue
        window_images.append(img_path)
        timestamp_labels.append("latest chunk")
    append_timestamped_image_list(
        content,
        window_images,
        fps=float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC),
        start_frame_index=window_start * FRAMES_PER_CHUNK,
        total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
        timestamp_labels=timestamp_labels,
        image_key="image_url",
        image_url_encoder=encode_image_base64,
    )

    request = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_observation"],
        "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
        "id": f"{video_id}_obs_{chunk_idx}",
    }
    return request


def build_observation_repair_request(
    chunk_idx: int,
    frame_paths: List[str],
    memory: MemoryState,
    video_id: str,
    *,
    stale_text: str = "",
) -> Dict:
    """Build a current-chunk timestamped-image repair request."""
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    chunk_frame_paths = [
        p for p in get_chunk_frame_paths(frame_paths, chunk_idx)
        if Path(p).exists()
    ]
    fps = float(FRAMES_PER_CHUNK / AGENT_CHUNK_SEC)

    recent_text = memory.format_recent_for_repair_prompt(limit=8)

    prompt = OBSERVATION_REPAIR_PROMPT.format(
        recent_thinks=recent_text,
        stale_text=(stale_text or "").strip()[:600] or "(none)",
        start=int(start),
        end=int(end),
        n_frames=len(chunk_frame_paths),
        fps=fps,
    )

    content: List[Dict] = [{"type": "text", "text": prompt}]
    append_timestamped_image_list(
        content,
        chunk_frame_paths,
        fps=fps,
        start_frame_index=chunk_idx * FRAMES_PER_CHUNK,
        total_num_frames=(chunk_idx + 1) * FRAMES_PER_CHUNK,
        context_label="latest chunk",
        image_key="image_url",
        image_url_encoder=encode_image_base64,
    )

    request = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_observation"],
        "temperature": min(float(PASS_CONFIG["pass2_rollout"]["temperature"]), 0.2),
        "id": f"{video_id}_obs_repair_{chunk_idx}",
    }
    return request


def parse_observation_result(raw: Optional[str]) -> str:
    """Parse observation output."""
    if raw is None:
        return "Scene continues without notable changes."
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = raw.strip('"').strip("'").strip()
    lines = []
    for line in raw.splitlines():
        if _FRAME_TAG_LINE_RE.fullmatch(line):
            continue
        lines.append(line)
    raw = "\n".join(lines).strip()
    raw = _FRAME_TAG_INLINE_RE.sub(" ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if (
        not raw
        or _PLACEHOLDER_TEXT_RE.fullmatch(raw)
        or _META_REASONING_RE.search(raw)
        or "the user wants" in raw[:160].lower()
    ):
        return "Scene continues without notable changes."
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


def _item_token_count(item: Dict) -> int:
    text = item.get("text", "")
    tokenizer = get_tokenizer()
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(len(text) // 4, 1)


def _item_time_bounds(item: Dict) -> Tuple[int, int]:
    if item.get("type") == "think":
        chunk = int(item.get("chunk", 0))
        return (
            int(chunk * AGENT_CHUNK_SEC),
            int(chunk * AGENT_CHUNK_SEC + AGENT_CHUNK_SEC),
        )
    tr = item.get("time_range") or []
    if isinstance(tr, list) and len(tr) == 2:
        return int(tr[0]), int(tr[1])
    return 0, 0


def _chunks_from_time_range(time_range: List[int]) -> List[int]:
    if not (isinstance(time_range, list) and len(time_range) == 2):
        return []
    try:
        start_s, end_s = float(time_range[0]), float(time_range[1])
    except (TypeError, ValueError):
        return []
    if end_s <= start_s:
        return []
    start_chunk = int(start_s / float(AGENT_CHUNK_SEC))
    # time_range end is exclusive; subtract a tiny epsilon before flooring.
    end_chunk = int((end_s - 1e-6) / float(AGENT_CHUNK_SEC))
    return list(range(start_chunk, end_chunk + 1))


def _item_source_chunks(item: Dict) -> List[int]:
    if item.get("type") == "think":
        try:
            return [int(item["chunk"])]
        except (KeyError, TypeError, ValueError):
            return []
    chunks = []
    for c in item.get("source_chunks") or []:
        try:
            chunks.append(int(c))
        except (TypeError, ValueError):
            continue
    if chunks:
        return sorted(set(chunks))
    return _chunks_from_time_range(item.get("time_range") or [])


def _range_source_chunks(items: List[Dict]) -> List[int]:
    chunks: List[int] = []
    for item in items:
        chunks.extend(_item_source_chunks(item))
    return sorted(set(chunks))


def _range_time_bounds(items: List[Dict]) -> Tuple[int, int]:
    bounds = [_item_time_bounds(item) for item in items]
    starts = [b[0] for b in bounds]
    ends = [b[1] for b in bounds]
    return (min(starts), max(ends)) if bounds else (0, 0)


_MEM_BLOCK_RE = re.compile(r"<MEM>\s*(.*?)\s*</MEM>", re.DOTALL | re.IGNORECASE)
_MEM_LINE_RE = re.compile(
    r'<m\s+t="(\d+)(?:\s*-\s*(\d+))?"\s*>(.*?)</m>',
    re.DOTALL | re.IGNORECASE,
)


def _xml_text(text: object) -> str:
    """Escape text embedded inside XML-ish teacher prompt tags."""
    return html.escape(str(text or "").strip(), quote=False)


def _format_compact_memory_block(items: List[Dict]) -> str:
    """Render current summary items as OLD_MEMORY for compact update."""
    lines = ["<MEM>"]
    for item in items:
        if item.get("type") != "summary":
            continue
        start, end = _item_time_bounds(item)
        if end < start:
            start, end = end, start
        lines.append(f'  <m t="{int(start)}-{int(end)}">{_xml_text(item.get("text", ""))}</m>')
    lines.append("</MEM>")
    return "\n".join(lines)


def _format_new_captions_block(items: List[Dict]) -> str:
    """Render raw recent think items as NEW_CAPTIONS for compact update."""
    lines = ["<NEW_CAPTIONS>"]
    for item in items:
        if item.get("type") != "think":
            continue
        try:
            chunk = int(item.get("chunk"))
        except (TypeError, ValueError):
            start, _end = _item_time_bounds(item)
            chunk = int(start)
        lines.append(f'  <c t="{chunk}">{_xml_text(item.get("text", ""))}</c>')
    lines.append("</NEW_CAPTIONS>")
    return "\n".join(lines)


def _extract_mem_block(raw: str) -> str:
    """Return a normalized ``<MEM>`` block from a teacher response.

    Large teacher models often obey the semantic task but drift on the thin XML
    wrapper: either returning bare ``<m>`` lines without ``<MEM>``, or omitting
    ``</m>`` on each line while still separating entries line-by-line. Repair
    those mechanical forms here so useful compact memories do not fall back to
    deterministic summaries.
    """
    text = re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.DOTALL).strip()
    if not text:
        return ""

    mem_match = _MEM_BLOCK_RE.search(text)
    body = mem_match.group(1) if mem_match else text
    if not re.search(r"<m\b", body, flags=re.IGNORECASE):
        return mem_match.group(0).strip() if mem_match else ""

    line_re = re.compile(
        r"(<m\b[^>]*>)(.*?)(?=(?:\n\s*<m\b)|(?:\s*</MEM>)|$)",
        re.DOTALL | re.IGNORECASE,
    )
    lines: List[str] = []
    for match in line_re.finditer(body):
        open_tag = match.group(1).strip()
        payload = match.group(2)
        payload = re.sub(r"</?MEM>", "", payload, flags=re.IGNORECASE)
        if "</m>" in payload.lower():
            payload = re.split(r"</m>", payload, maxsplit=1, flags=re.IGNORECASE)[0]
        payload = re.sub(r"\s+", " ", payload).strip()
        if payload:
            lines.append(f"  {open_tag}{payload}</m>")
    if not lines:
        return mem_match.group(0).strip() if mem_match else ""
    return "<MEM>\n" + "\n".join(lines) + "\n</MEM>"


def parse_compact_memory_entries(raw: str) -> List[Dict]:
    """Parse teacher/user <MEM> block into summary-like timeline entries."""
    block = _extract_mem_block(raw or "")
    if not block:
        return []
    entries: List[Dict] = []
    for m in _MEM_LINE_RE.finditer(block):
        try:
            start = int(m.group(1))
            end = int(m.group(2) if m.group(2) is not None else m.group(1))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        text = html.unescape(re.sub(r"\s+", " ", m.group(3)).strip())
        if not _is_valid_compress_text(text):
            continue
        entries.append({
            "type": "summary",
            "time_range": [start, end],
            "text": text,
            "source_chunks": list(range(start, end + 1)),
            "merge_level": 1,
            "compact_memory": True,
        })
    entries.sort(key=lambda item: (item["time_range"][0], item["time_range"][1]))
    return entries


def _entries_to_mem_text(entries: List[Dict]) -> str:
    lines = ["<MEM>"]
    for entry in entries:
        tr = entry.get("time_range") or [0, 0]
        lines.append(f'  <m t="{int(tr[0])}-{int(tr[1])}">{entry.get("text", "").strip()}</m>')
    lines.append("</MEM>")
    return "\n".join(lines)


# Compression scoring weights (configurable, sum ≈ 1.0)
COMPRESS_W_CONTENT  = 0.30  # content importance → avoid compressing
COMPRESS_W_MERGE    = 0.20  # re-compression penalty → avoid
COMPRESS_W_BOUNDARY = 0.15  # event boundary → avoid splitting
COMPRESS_W_RECENCY  = 0.20  # time recency → prefer compressing old
COMPRESS_W_TOKEN    = 0.15  # token saving → prefer compressing more tokens


def _item_importance(item: Dict, ev_index: Dict) -> float:
    """Per-item importance score [0, 1].

    Normalized: each factor is [0, 1], weighted sum capped at 1.0.
    """
    if item.get("type") == "summary":
        has_digits = 1.0 if any(c.isdigit() for c in item.get("text", "")) else 0.0
        return min(1.0, has_digits * 0.5)

    # Think: check evidence
    cap = ev_index.get(item.get("chunk"))
    if not cap:
        return 0.0

    entity_score = min(len(cap.get("visible_entities", [])) / 5.0, 1.0)
    ocr_score = 1.0 if cap.get("ocr") else 0.0
    digit_score = 1.0 if any(
        any(c.isdigit() for c in f.get("fact", ""))
        for f in cap.get("atomic_facts", [])
        if f.get("confidence", 0) >= CONFIDENCE_THRESHOLD
    ) else 0.0
    change_score = min(len(cap.get("state_changes", [])), 2) / 2.0

    return min(1.0, entity_score * 0.25 + ocr_score * 0.35 + digit_score * 0.20 + change_score * 0.20)


def score_range_for_compression(
    items: List[Dict],
    start_idx: int,
    timeline_len: int,
    evidence: Optional[List[Dict]] = None,
    *,
    range_tokens: Optional[int] = None,
    total_timeline_tokens: Optional[int] = None,
) -> float:
    """Score a candidate range for compression. LOWER = better to compress.

    All dimensions normalized to [0, 1], then weighted.
    Total score range: approximately [-W_TOKEN, W_CONTENT+W_MERGE+W_BOUNDARY+W_RECENCY].
    """
    ev_index = _evidence_by_chunk(evidence) if evidence else {}
    n = len(items)

    # --- content [0, 1]: average per-item importance ---
    if n > 0:
        content = sum(_item_importance(item, ev_index) for item in items) / n
    else:
        content = 0.0

    # --- merge [0, 1]: max merge_level / 3 (level 3+ → 1.0) ---
    max_level = max((item.get("merge_level", 0) for item in items), default=0)
    merge = min(max_level / 3.0, 1.0)

    # --- boundary [0, 1]: state_change at first/last item ---
    boundary = 0.0
    if ev_index:
        for edge_item in [items[0], items[-1]]:
            if edge_item.get("type") == "think":
                cap = ev_index.get(edge_item.get("chunk"))
                if cap and cap.get("state_changes"):
                    boundary += 0.5  # max 1.0 if both edges have state_change

    # --- recency [0, 1]: center position in timeline ---
    if timeline_len > 1:
        recency = (start_idx + n / 2) / timeline_len
    else:
        recency = 0.5

    # --- token_ratio [0, 1]: true fraction of visible-memory tokens ---
    if range_tokens is None:
        range_tokens = sum(_item_token_count(item) for item in items)
    if total_timeline_tokens is None:
        total_timeline_tokens = range_tokens
    token_ratio = min(1.0, range_tokens / max(total_timeline_tokens, 1))

    # --- Weighted combination ---
    score = (
        COMPRESS_W_CONTENT  * content       # high content → high score → avoid
        + COMPRESS_W_MERGE  * merge         # has summary → high score → avoid
        + COMPRESS_W_BOUNDARY * boundary    # event boundary → high score → avoid
        + COMPRESS_W_RECENCY * recency      # recent → high score → avoid
        - COMPRESS_W_TOKEN  * token_ratio   # more tokens → low score → prefer
    )
    return score


def choose_optimal_compress_range(
    timeline: List[Dict],
    evidence: Optional[List[Dict]] = None,
) -> Tuple[List[int], Dict]:
    """Choose the best contiguous visible-memory range to compress.

    Trigger timing and range selection both run over the unified visible
    timeline, so older summaries can be re-compressed together with adjacent
    raw observations. Summaries receive a soft merge_level penalty, not a hard
    exclusion.

    Returns: (selected_indices in timeline, policy_meta)
    """
    n = len(timeline)
    best_indices = None
    best_score = float("inf")
    token_counts = [_item_token_count(item) for item in timeline]
    token_prefix = [0]
    for count in token_counts:
        token_prefix.append(token_prefix[-1] + count)
    total_tokens = token_prefix[-1]

    # Enumerate contiguous ranges in visible-memory order. A valid range must
    # contain some raw thinks; compressing only summaries compounds loss without
    # reducing recent_thinks pressure.
    for size in range(COMPRESS_RANGE_MIN, min(COMPRESS_RANGE_MAX + 1, n + 1)):
        for start in range(0, n - size + 1):
            candidate_indices = list(range(start, start + size))
            candidate_items = [timeline[i] for i in candidate_indices]
            n_thinks = sum(
                1 for item in candidate_items if item.get("type") == "think"
            )
            if n_thinks < 2:
                continue
            score = score_range_for_compression(
                candidate_items,
                start,
                n,
                evidence,
                range_tokens=token_prefix[start + size] - token_prefix[start],
                total_timeline_tokens=total_tokens,
            )
            if score < best_score:
                best_score = score
                best_indices = candidate_indices

    if best_indices is None:
        think_positions = [
            i for i, item in enumerate(timeline) if item.get("type") == "think"
        ]
        if len(think_positions) >= COMPRESS_RANGE_MIN:
            best_indices = think_positions[:COMPRESS_RANGE_MIN]
        else:
            best_indices = think_positions

    meta = {
        "score": round(best_score, 2) if best_score < float("inf") else -1,
        "range_indices": best_indices,
        "range_size": len(best_indices) if best_indices else 0,
        "timeline_size": len(timeline),
        "recent_thinks_size": sum(1 for item in timeline if item.get("type") == "think"),
        "n_thinks_in_range": sum(1 for i in (best_indices or []) if timeline[i].get("type") == "think"),
        "n_summaries_in_range": sum(1 for i in (best_indices or []) if timeline[i].get("type") == "summary"),
        "source_chunks_in_range": _range_source_chunks(
            [timeline[i] for i in (best_indices or [])]
        ),
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

    Compression summaries are generated from text memory, not from fresh
    video frames. Student/runtime compression prompts are also text-only. The
    `frame_paths` arg is kept for backward compatibility with callers and is
    no longer consumed here.
    """
    if COMPACT_MEMORY_UPDATE_MODE:
        if not pre_action_timeline:
            return None
        raw_think_items = [item for item in pre_action_timeline if item.get("type") == "think"]
        if not raw_think_items:
            return None
        selected_indices = list(range(len(pre_action_timeline)))
        source_items = [pre_action_timeline[i] for i in selected_indices]
        first_time, last_time = _range_time_bounds(source_items)
        raw_think_chunks = [
            int(item["chunk"]) for item in raw_think_items
            if isinstance(item.get("chunk"), int) or str(item.get("chunk", "")).isdigit()
        ]
        compress_chunks = _range_source_chunks(source_items)
        merge_level = (
            max((int(item.get("merge_level", 0)) for item in source_items), default=0)
            + 1
        )
        old_memory_text = _format_compact_memory_block(source_items)
        new_captions_text = _format_new_captions_block(source_items)
        prompt = COMPACT_MEMORY_UPDATE_PROMPT.format(
            old_memory=old_memory_text,
            new_captions=new_captions_text,
            start=int(first_time),
            end=int(last_time),
        )
        return {
            "messages": [
                {"role": "system", "content": COMPACT_MEMORY_UPDATE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": PASS_CONFIG["pass2_rollout"]["max_tokens_compress"],
            "temperature": PASS_CONFIG["pass2_rollout"]["temperature"],
            "id": f"{video_id}_compact_mem_{chunk_idx}",
            "_meta": {
                "task_type": "compact_memory_update",
                "time_range": [int(first_time), int(last_time)],
                "selected_indices": selected_indices,
                "chunks": compress_chunks,
                "raw_think_chunks": raw_think_chunks,
                "merge_level": merge_level,
                "teacher_policy": {
                    "mode": "compact_memory_update",
                    "timeline_size": len(pre_action_timeline),
                    "n_new_captions": len(raw_think_items),
                    "n_old_memory": sum(1 for item in pre_action_timeline if item.get("type") == "summary"),
                },
                "overlap_chunks": [],
                "has_visual_context": False,
                "observations_text": "\n".join(
                    MemoryState._format_timeline_item_as_memory_tag(item)
                    for item in source_items
                ),
                "old_memory_text": old_memory_text,
                "new_captions_text": new_captions_text,
            },
        }

    selected_indices, policy_meta = choose_optimal_compress_range(
        pre_action_timeline, evidence
    )

    if not selected_indices:
        return None

    to_compress = [pre_action_timeline[i] for i in selected_indices]

    obs_lines = []
    for item in to_compress:
        obs_lines.append(MemoryState._format_timeline_item_as_memory_tag(item))
    obs_text = "\n".join(obs_lines)

    first_time, last_time = _range_time_bounds(to_compress)

    target_length = estimate_summary_length(to_compress, evidence)

    raw_think_chunks = [
        int(item["chunk"]) for item in to_compress if item.get("type") == "think"
    ]
    compress_chunks = _range_source_chunks(to_compress)
    merge_level = (
        max((int(item.get("merge_level", 0)) for item in to_compress), default=0)
        + 1
    )

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
            "selected_indices": selected_indices,
            "chunks": compress_chunks,
            "raw_think_chunks": raw_think_chunks,
            "merge_level": merge_level,
            "teacher_policy": policy_meta,
            "overlap_chunks": [],
            "has_visual_context": False,
            "observations_text": obs_text,
        },
    }


def _fallback_compress_text(meta: Dict) -> str:
    """Extract a deterministic summary fallback from the selected observations."""
    obs = meta.get("observations_text", "")
    tagged_texts = []
    for m in re.finditer(
        r"<(?:memory_think|compressed)>(.*?)</(?:memory_think|compressed)>",
        obs,
        flags=re.DOTALL,
    ):
        try:
            payload = json.loads(m.group(1))
        except (TypeError, json.JSONDecodeError):
            continue
        text = str(payload.get("text", "")).strip()
        if text:
            tagged_texts.append(text)
    if tagged_texts:
        obs = " ".join(tagged_texts)
    obs = re.sub(r"</?(?:summary|memory_think|compressed)[^>]*>", " ", obs)
    obs = re.sub(r"\[[^\]]+\]\s*", " ", obs)
    obs = " ".join(obs.split())
    if not obs:
        return "Observations recorded during this period."
    if len(obs) > 700:
        obs = obs[:700].rsplit(" ", 1)[0]
    return obs


def _fallback_compact_entries(meta: Dict, target_lines: int = 5) -> List[Dict]:
    """Deterministic compact-memory fallback when teacher output is invalid."""
    records: List[Dict] = []
    obs = meta.get("observations_text", "")
    for m in re.finditer(
        r"<(memory_think|compressed)>(.*?)</\1>",
        obs,
        flags=re.DOTALL,
    ):
        kind = m.group(1)
        try:
            payload = json.loads(m.group(2))
        except (TypeError, json.JSONDecodeError):
            continue
        text = str(payload.get("text", "")).strip()
        if not text:
            continue
        if kind == "memory_think":
            time_raw = str(payload.get("time", "")).strip()
            nums = [int(x) for x in re.findall(r"\d+", time_raw)]
            if nums:
                start, end = nums[0], nums[-1]
            else:
                start, end = meta.get("time_range", [0, 0])
        else:
            tr = payload.get("time_range") or []
            if isinstance(tr, list) and len(tr) == 2:
                start, end = int(tr[0]), int(tr[1])
            else:
                start, end = meta.get("time_range", [0, 0])
        if end < start:
            start, end = end, start
        records.append({"start": int(start), "end": int(end), "text": text})

    if not records:
        start, end = meta.get("time_range", [0, 0])
        records = [{
            "start": int(start),
            "end": int(end),
            "text": _fallback_compress_text(meta),
        }]

    records.sort(key=lambda r: (r["start"], r["end"]))
    if len(records) <= 6:
        groups = [[r] for r in records]
    else:
        n_groups = max(4, min(6, int(target_lines)))
        groups = []
        for gi in range(n_groups):
            lo = round(gi * len(records) / n_groups)
            hi = round((gi + 1) * len(records) / n_groups)
            groups.append(records[lo:hi])

    entries: List[Dict] = []
    for group in groups:
        group = [r for r in group if r]
        if not group:
            continue
        start = min(r["start"] for r in group)
        end = max(r["end"] for r in group)
        text = " ".join(r["text"] for r in group)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 260:
            text = text[:260].rsplit(" ", 1)[0].rstrip(".;,") + "."
        entries.append({
            "type": "summary",
            "time_range": [start, end],
            "text": text,
            "source_chunks": list(range(start, end + 1)),
            "merge_level": int(meta.get("merge_level", 1) or 1),
            "compact_memory": True,
        })
    return entries[:6]


def _is_valid_compress_text(text: object) -> bool:
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in {"...", "…", "<summary text here>", "summary text here"}:
        return False
    if _PLACEHOLDER_TEXT_RE.fullmatch(stripped):
        return False
    if _META_REASONING_RE.search(stripped):
        return False
    return True


def parse_compress_result(raw: Optional[str], meta: Dict) -> Dict:
    """Parse compression summary output."""
    source_chunks = sorted(int(c) for c in (meta.get("chunks") or []))
    merge_level = int(meta.get("merge_level", 1) or 1)
    if meta.get("task_type") == "compact_memory_update":
        fallback_entries = _fallback_compact_entries(meta)
        default = {
            "time_range": meta["time_range"],
            "text": _entries_to_mem_text(fallback_entries),
            "entries": fallback_entries,
            "source_chunks": source_chunks,
            "merge_level": merge_level,
            "parse_success": False,
            "compact_memory_update": True,
        }
        if raw is None:
            return default
        raw_no_think = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        literal_mem = _MEM_BLOCK_RE.search(raw_no_think)
        mem_block = _extract_mem_block(raw_no_think)
        entries = parse_compact_memory_entries(raw_no_think)
        if not entries:
            default["_raw"] = raw_no_think[:4000]
            return default
        n_entries = len(entries)
        time_range = [
            min(int(e["time_range"][0]) for e in entries),
            max(int(e["time_range"][1]) for e in entries),
        ]
        parse_success = 4 <= n_entries <= 6
        out = {
            "time_range": time_range,
            # Keep the teacher's raw <MEM> block as the canonical text
            # artifact. Parsed <m t="..."> entries are diagnostics/state for
            # the rollout; downstream trajectory rendering reuses this whole
            # block instead of rebuilding memory from the parsed lines.
            "text": mem_block or _entries_to_mem_text(entries),
            "entries": entries,
            "source_chunks": sorted(set(c for e in entries for c in e.get("source_chunks", []))),
            "merge_level": merge_level,
            "parse_success": parse_success,
            "compact_memory_update": True,
            "n_entries": n_entries,
        }
        literal_block = literal_mem.group(0).strip() if literal_mem else raw_no_think
        raw_open_m = len(re.findall(r"<m\b", literal_block, flags=re.IGNORECASE))
        raw_close_m = len(re.findall(r"</m>", literal_block, flags=re.IGNORECASE))
        if (not literal_mem) or raw_open_m != raw_close_m:
            out["format_repaired"] = True
        if not parse_success:
            out["_raw"] = raw_no_think[:4000]
            out["parse_warning"] = "expected_4_to_6_entries"
        return out

    default = {
        "time_range": meta["time_range"],
        "text": _fallback_compress_text(meta),
        "source_chunks": source_chunks,
        "merge_level": merge_level,
        "parse_success": False,
    }

    if raw is None:
        return default

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
        text = parsed.get("text", "")
        if not _is_valid_compress_text(text):
            default["_raw"] = raw[:4000]
            return default
        return {
            "time_range": meta["time_range"],
            "text": text.strip(),
            "source_chunks": source_chunks,
            "merge_level": merge_level,
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
                            text = parsed.get("text", "")
                            if not _is_valid_compress_text(text):
                                default["_raw"] = raw[:4000]
                                return default
                            return {
                                "time_range": meta["time_range"],
                                "text": text.strip(),
                                "source_chunks": source_chunks,
                                "merge_level": merge_level,
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
    chunk_log_path: Optional[Path] = None,
) -> Dict:
    """Run question-blind streaming rollout for a single video.

    Args:
        chunk_log_path: If set, appends a JSONL line per chunk for real-time debug.
                        Contains: chunk_idx, think text, timeline state, compression info.
                        Does NOT affect final output (snapshots/thinks/events all unchanged).
    """
    memory = MemoryState()
    thinks = []
    compression_events = []
    snapshots = {}
    enable_thinking = bool(PASS_CONFIG["pass2_rollout"].get("thinking", False))
    evidence_by_chunk: Dict[int, Dict] = {}
    if evidence:
        for i, cap in enumerate(evidence):
            if not isinstance(cap, dict):
                continue
            try:
                cidx = int(cap.get("chunk_idx", i))
            except (TypeError, ValueError):
                cidx = i
            evidence_by_chunk[cidx] = cap
    planned_compact_intervals: List[int] = []
    planned_compact_boundaries: List[int] = []
    planned_compact_boundary_set = set()
    if COMPACT_MEMORY_UPDATE_MODE and COMPACT_MEMORY_BALANCE_SEGMENTS:
        planned_compact_intervals = plan_compact_memory_intervals(num_chunks)
        planned_compact_boundaries = plan_compact_memory_boundaries(num_chunks)
        planned_compact_boundary_set = set(planned_compact_boundaries)

    for chunk_idx in range(num_chunks):
        # --- 1. Snapshot BEFORE this step's think ---
        snapshots[chunk_idx] = memory.snapshot(chunk_idx)
        pre_action_timeline = snapshots[chunk_idx]["timeline"]
        pre_action_thinks = snapshots[chunk_idx]["recent_thinks"]

        raw_trigger_diag = memory.compress_trigger_diagnostic()
        compress_trigger_diag = raw_trigger_diag
        if COMPACT_MEMORY_UPDATE_MODE and COMPACT_MEMORY_BALANCE_SEGMENTS:
            schedule_ready = chunk_idx in planned_compact_boundary_set
            compress_trigger_diag = dict(raw_trigger_diag)
            compress_trigger_diag.update({
                "triggered": schedule_ready,
                "mode": "compact_memory_update_balanced",
                "schedule_ready": schedule_ready,
                "planned_intervals": planned_compact_intervals,
                "planned_boundaries": planned_compact_boundaries,
                "reason": (
                    "balanced_boundary" if schedule_ready
                    else (
                        "balanced_wait_after_token_ready"
                        if raw_trigger_diag.get("token_ready")
                        else "balanced_wait"
                    )
                ),
            })
            should_compress_now = bool(schedule_ready)
        elif COMPACT_MEMORY_UPDATE_MODE:
            should_compress_now = bool(compress_trigger_diag.get("triggered"))
        else:
            should_compress_now = (
                bool(compress_trigger_diag.get("triggered"))
                and len(pre_action_timeline) >= COMPRESS_RANGE_MIN
                and len(pre_action_thinks) >= 2
            )

        # --- 2. Get think for current chunk ---
        # v12.25: production path uses pass1's independent current-only
        # `think` observation-note field. This is NOT model reasoning mode.
        # It removes the old pass2 teacher-observation call that saw full
        # text memory and could stale-repeat history.
        repaired = False
        repair_attempted = False
        repair_rejected = False
        repair_meta: Dict = {}
        visual_delta_mse = None
        think_source = "pass2_teacher_observation"

        cap = evidence_by_chunk.get(chunk_idx)
        if cap is not None:
            think_text = build_think_from_pass1_evidence(cap)
            think_source = think_source_for_evidence(cap)
        else:
            logger.warning(
                "  [%s] missing pass1 evidence at chunk %d; falling back to "
                "legacy pass2 teacher observation",
                video_id,
                chunk_idx,
            )
            request = build_observation_request(chunk_idx, frame_paths, memory, video_id)
            # v12.11 hotfix: cap max_tokens client-side so long-memory chunks
            # don't trip vLLM's "max_tokens must be at least 1, got -<N>" error.
            safe_obs_max = _safe_max_tokens_for_pass2(request, request["max_tokens"])
            # v12.12: pass2 uses RUNTIME profile — same smart_resize bounds as
            # student inference, so teacher and student see identical visual
            # token sequences at every chunk (training-inference parity).
            mm_kwargs = dict(RUNTIME_MM_PROCESSOR_KWARGS)
            mm_kwargs["do_sample_frames"] = False
            raw = await client._call_one(
                messages=request["messages"],
                max_tokens=safe_obs_max,
                temperature=request["temperature"],
                request_id=request["id"],
                enable_thinking=enable_thinking,
                mm_processor_kwargs=mm_kwargs,
                media_io_kwargs=request.get("media_io_kwargs"),
            )
            think_text = parse_observation_result(raw)
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
                repair_request = build_observation_repair_request(
                    chunk_idx, frame_paths, memory, video_id, stale_text=think_text,
                )
                safe_repair_max = _safe_max_tokens_for_pass2(
                    repair_request, repair_request["max_tokens"],
                )
                try:
                    repair_raw = await client._call_one(
                        messages=repair_request["messages"],
                        max_tokens=safe_repair_max,
                        temperature=repair_request["temperature"],
                        request_id=repair_request["id"],
                        enable_thinking=enable_thinking,
                        mm_processor_kwargs=mm_kwargs,
                        media_io_kwargs=repair_request.get("media_io_kwargs"),
                    )
                    repaired_text = parse_observation_result(repair_raw)
                    if _is_repair_better(
                        repaired_text,
                        think_text,
                        memory.recent_thinks,
                    ):
                        think_text = repaired_text
                        repaired = True
                        think_source = "pass2_teacher_repair"
                        logger.info(
                            "  [%s] Repaired stale pass2 think at chunk %d: %s",
                            video_id, chunk_idx, repair_meta.get("reason", ""),
                        )
                    else:
                        repair_rejected = True
                        logger.warning(
                            "  [%s] Rejected stale pass2 repair at chunk %d: still too similar (%s)",
                            video_id, chunk_idx, repair_meta.get("reason", ""),
                        )
                except Exception as exc:
                    logger.warning(
                        "  [%s] Stale pass2 repair failed at chunk %d: %s",
                        video_id, chunk_idx, exc,
                    )

        think_record = {
            "chunk_idx": chunk_idx,
            "time": [chunk_idx * AGENT_CHUNK_SEC, (chunk_idx + 1) * AGENT_CHUNK_SEC],
            "think": think_text,
            "source": think_source,
        }
        if repaired:
            think_record["repair"] = repair_meta
        thinks.append(think_record)

        # --- 3. Compress (between timesteps) then append current think ---
        if should_compress_now:
            comp_request = build_compress_request(
                pre_action_timeline, memory, video_id, chunk_idx,
                evidence=evidence, frame_paths=frame_paths,
            )
            if comp_request is None:
                memory.add_think(chunk_idx, think_text)
                continue
            # v12.11 hotfix: same client-side cap for compress request.
            safe_comp_max = _safe_max_tokens_for_pass2(
                comp_request, comp_request["max_tokens"],
            )
            # compress teacher request is text-memory only; mm_processor_kwargs
            # is unused but harmless if passed.
            try:
                comp_raw = await client._call_one(
                    messages=comp_request["messages"],
                    max_tokens=safe_comp_max,
                    temperature=comp_request["temperature"],
                    request_id=comp_request["id"],
                    enable_thinking=enable_thinking,
                )
            except TruncatedCompletionError as exc:
                logger.warning(
                    "  [%s] Compression response truncated at chunk %d; "
                    "using deterministic fallback summary: %s",
                    video_id,
                    chunk_idx,
                    exc,
                )
                comp_raw = None
            summary = parse_compress_result(comp_raw, comp_request["_meta"])
            selected_indices = comp_request["_meta"]["selected_indices"]

            if summary.get("compact_memory_update") and summary.get("entries"):
                memory.replace_with_compact_entries(summary)
            else:
                memory.compress(summary, selected_indices=selected_indices)
            memory.add_think(chunk_idx, think_text)

            post_compress_tokens = memory.count_recent_tokens()
            hysteresis_ok = post_compress_tokens <= COMPRESS_HYSTERESIS_THRESHOLD
            if not hysteresis_ok:
                logger.warning(
                    f"  [{video_id}] Compression hysteresis violated at chunk {chunk_idx}: "
                    f"post-compress visible memory {post_compress_tokens} tok > "
                    f"{COMPRESS_HYSTERESIS_THRESHOLD} threshold"
                )

            compression_events.append({
                "trigger_chunk": chunk_idx,
                "summary": summary,
                "selected_indices": selected_indices,
                "compressed_thinks_chunks": comp_request["_meta"].get("chunks", []),
                "compressed_raw_think_chunks": comp_request["_meta"].get("raw_think_chunks", []),
                "memory_update_input": comp_request["messages"][-1]["content"],
                "old_memory_text": comp_request["_meta"].get("old_memory_text", ""),
                "new_captions_text": comp_request["_meta"].get("new_captions_text", ""),
                "teacher_policy": comp_request["_meta"].get("teacher_policy", {}),
                "trigger_diagnostic": compress_trigger_diag,
                "compact_memory_update": bool(summary.get("compact_memory_update")),
                "hysteresis_ok": hysteresis_ok,
                "post_compress_tokens": post_compress_tokens,
            })
            logger.debug(f"  [{video_id}] Compress at chunk {chunk_idx}: {summary['time_range']}")
        else:
            memory.add_think(chunk_idx, think_text)

        # --- Per-chunk debug log (real-time, does not affect final output) ---
        if chunk_log_path:
            log_entry = {
                "video_id": video_id,
                "chunk": chunk_idx,
                "t": f"{chunk_idx * AGENT_CHUNK_SEC}-{(chunk_idx+1) * AGENT_CHUNK_SEC}",
                "think": think_text[:120],
                "timeline_len": len(memory.timeline),
                "n_thinks": len(memory.recent_thinks),
                "n_summaries": len(memory.compressed_segments),
                "tokens": memory.count_tokens(),
                "compressed": should_compress_now,
            }
            if repair_attempted:
                log_entry["repair_attempted"] = True
                log_entry["repair_trigger_reason"] = repair_meta.get("reason")
                log_entry["repair_run_length"] = repair_meta.get("run_length")
                log_entry["repair_evidence_drift"] = repair_meta.get("evidence_drift")
            if visual_delta_mse is not None:
                log_entry["visual_delta_mse"] = round(visual_delta_mse, 3)
            if repaired:
                log_entry["repaired"] = True
                log_entry["repair_reason"] = repair_meta.get("reason")
            elif repair_rejected:
                log_entry["repair_rejected"] = True
                log_entry["repair_rejected_reason"] = "still_too_similar"
            if should_compress_now and compression_events and compression_events[-1]["trigger_chunk"] == chunk_idx:
                ce = compression_events[-1]
                log_entry["compress_range"] = ce["summary"].get("time_range")
                log_entry["compress_score"] = ce["teacher_policy"].get("score")
                log_entry["post_tokens"] = ce.get("post_compress_tokens")
                log_entry["summary_preview"] = ce["summary"].get("text", "")[:80]
            with open(chunk_log_path, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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
        "compact_memory_plan": {
            "enabled": bool(COMPACT_MEMORY_UPDATE_MODE and COMPACT_MEMORY_BALANCE_SEGMENTS),
            "min_chunks": COMPACT_MEMORY_MIN_NEW_CHUNKS,
            "target_chunks": COMPACT_MEMORY_TARGET_NEW_CHUNKS,
            "max_chunks": COMPACT_MEMORY_MAX_NEW_CHUNKS,
            "intervals": planned_compact_intervals,
            "boundaries": planned_compact_boundaries,
        },
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
    from .cache_version import stage_version_ok
    if not stage_version_ok("2"):
        return None
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
