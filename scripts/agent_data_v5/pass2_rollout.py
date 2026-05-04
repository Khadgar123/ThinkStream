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
from PIL import Image

from .config import (
    AGENT_CHUNK_SEC,
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
from .pass1a_evidence import get_chunk_frame_paths
from scripts.agent_data_pipeline.vllm_client import encode_image_base64
from thinkstream.data.agent_protocol import append_timestamped_image_list

logger = logging.getLogger(__name__)


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
# Conservative input estimate per pass2 observation request (v12.12):
#   visual:  32 frames × ~235 tok/frame = ~7,520 tok (RUNTIME mm_processor_kwargs)
#   memory:  recent_thinks ≤ 4000 tok + compressed ≤ 1400 tok = ~5400 tok
#   prompt template + safety: ~700 tok
#   ─────────────────────────────────────────────────
#   estimated input ~13,600 tok worst case → max_tokens=16K fits in 64K cap.
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

    v12.18 (2026-05-04): observation requests send the full sliding visual
    window as a timestamped image list, ordered from older context to the
    latest chunk. Real A/B calls on stale pass2 failures showed that this is
    more reliable than the vLLM video_url path for making the latest chunk
    win against stale text memory, while still preserving the student's visual
    sliding-window distribution.
    """
    start = chunk_idx * AGENT_CHUNK_SEC
    end = start + AGENT_CHUNK_SEC
    window_start = compute_visual_window_start(chunk_idx)

    memory_text = memory.format_for_prompt()

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
    for c in range(window_start, chunk_idx + 1):
        label = "latest chunk" if c == chunk_idx else "older context"
        for img_path in get_chunk_frame_paths(frame_paths, c):
            if not Path(img_path).exists():
                continue
            window_images.append(img_path)
            timestamp_labels.append(label)
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

    recent_lines = []
    for item in memory.recent_thinks[-8:]:
        recent_lines.append(f'[{item["time"]}] {item.get("text", "")}')
    recent_text = "\n".join(recent_lines) or "(none)"

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

    # --- token_ratio [0, 1]: fraction of total timeline tokens ---
    tokenizer = get_tokenizer()
    if tokenizer:
        range_tokens = sum(len(tokenizer.encode(item.get("text", ""), add_special_tokens=False)) for item in items)
    else:
        range_tokens = sum(len(item.get("text", "")) // 4 for item in items)
    # Estimate total timeline tokens (avoid recomputing full timeline)
    avg_item_tokens = max(range_tokens / max(n, 1), 1)
    est_total = avg_item_tokens * timeline_len
    token_ratio = range_tokens / max(est_total, 1)

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

    v12.12 (2026-05-01): compress is text-only. Student emits compress in
    inter-chunk shape C (agent_loop.py:812-816, pass5_messages.py:104+178)
    with NO visual_window and NO frames. Teacher must match that exact
    distribution — passing overlap frames here used to make teacher
    "verify entity details from video" but produced summaries the student
    cannot reproduce at inference (it has only memory text). The
    `frame_paths` arg is kept for backward compat with callers; it is
    no longer consumed.
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

    compress_chunks = [item["chunk"] for item in to_compress if item.get("type") == "think"]

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
            "teacher_policy": policy_meta,
            "overlap_chunks": [],
            "has_visual_context": False,
            "observations_text": obs_text,
        },
    }


def _fallback_compress_text(meta: Dict) -> str:
    """Extract a deterministic summary fallback from the selected observations."""
    obs = meta.get("observations_text", "")
    obs = re.sub(r"</?summary[^>]*>", " ", obs)
    obs = re.sub(r"\[[^\]]+\]\s*", " ", obs)
    obs = " ".join(obs.split())
    if not obs:
        return "Observations recorded during this period."
    if len(obs) > 700:
        obs = obs[:700].rsplit(" ", 1)[0]
    return obs


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
    default = {
        "time_range": meta["time_range"],
        "text": _fallback_compress_text(meta),
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
        # v12.11 hotfix: cap max_tokens client-side so long-memory chunks
        # don't trip vLLM's "max_tokens must be at least 1, got -<N>" error.
        safe_obs_max = _safe_max_tokens_for_pass2(request, request["max_tokens"])
        # v12.12: pass2 uses RUNTIME profile — same smart_resize bounds as
        # student inference, so teacher and student see identical visual
        # token sequences at every chunk (training-inference parity).
        enable_thinking = bool(PASS_CONFIG["pass2_rollout"].get("thinking", False))
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
            # compress request is text-only (v12.12 P0 dropped overlap frames),
            # mm_processor_kwargs unused but harmless if passed.
            comp_raw = await client._call_one(
                messages=comp_request["messages"],
                max_tokens=safe_comp_max,
                temperature=comp_request["temperature"],
                request_id=comp_request["id"],
                enable_thinking=enable_thinking,
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
