"""Streaming Agent Loop for inference.

Single-step inference loop that mirrors the data construction pipeline exactly:
- Maintains MemoryState (compressed_segments, recent_thinks, queries log)
- System-triggered compression (token-count threshold)
- Recall orchestration (parse query → retrieve → second generate)
- Constructs per-timestep input matching SFT training format

Each step is an independent single-turn inference (no KV cache reuse across steps).
This guarantees train/inference format identity.
"""

import json
import logging
import os
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Dict, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    RECALL_RETURN_CHUNKS,
    VISUAL_WINDOW_CHUNKS,
    build_recalled_frames_metadata,
    build_recall_result_metadata,
    build_recall_result_user_content,
    build_user_content,
    canonical_answer_instruction,
    format_memory_block,
    normalize_frame_protocol,
    normalize_render_layout,
    parse_agent_output_v12,
    recall_time_string_for_chunks,
    select_recall_chunks,
    system_prompt_for_frame_protocol,
    action_space_error_for_turn,
    append_query_answer_with_timing,
    query_is_complete,
    tools_for_turn,
)


def _parse_agent_output(output_text: str) -> Dict:
    """Parse v12 model output into the legacy {action, payload, think} shape.

    The agent loop logic keys on ``parsed["action"]`` (silent/response/recall/
    compress) and ``parsed["payload"]`` for back-compat with the surrounding
    orchestration code. We adapt the v12 parser (which emits ``kind`` +
    ``answer_text`` / ``tool_call``) into that shape here.
    """
    v12 = parse_agent_output_v12(output_text)
    out: Dict = {
        "raw": v12.get("raw", output_text),
        "raw_output": v12.get("raw", output_text),
        "think": v12.get("think", ""),
        "action": "",
        "payload": {},
    }
    kind = v12.get("kind", "unknown")
    if kind == "answer":
        text = v12.get("answer_text") or ""
        if text:
            out["action"] = "response"
            out["payload"]["response"] = text
        else:
            out["action"] = "silent"
    elif kind == "recall":
        out["action"] = "recall"
        tc = v12.get("tool_call") or {}
        args = tc.get("arguments") or {}
        out["payload"]["query"] = {
            "query": args.get("query", ""),
            "time_range": args.get("time_range", ""),
        }
    elif kind == "compress":
        out["action"] = "compress"
        tc = v12.get("tool_call") or {}
        args = tc.get("arguments") or {}
        out["payload"]["summary"] = {
            "time_range": args.get("time_range", []),
            "text": args.get("text", ""),
        }
    if v12.get("format_error"):
        out["format_error"] = v12["format_error"]
    return out

logger = logging.getLogger(__name__)

# Token-based compression trigger (matches data construction config.py).
# v12.5 (2026-04-29): 1s/chunk + 16K context → text-memory budget grows 4×
# so it exceeds visual horizon. See scripts/agent_data_v5/config.py docstring
# above RECENT_THINKS_TOKEN_BUDGET for the full 16K allocation breakdown.
RECENT_THINKS_TOKEN_BUDGET = 4000
COMPRESS_TRIGGER_RATIO = 0.8
COMPRESS_TOKEN_THRESHOLD = int(RECENT_THINKS_TOKEN_BUDGET * COMPRESS_TRIGGER_RATIO)  # 3200
COMPRESS_RANGE_MIN = 8              # ~8s of older thinks under 1s/chunk
COMPRESS_RANGE_MAX = 24             # ~24s, sized for 4000-token budget
COMPRESS_REMOVE_TOKENS = 1500       # ~40% budget eviction per compress
SUMMARY_TOKENS_MAX = 280            # matches config.py


def select_compress_range_by_tokens(
    thinks: List[Dict],
    token_count_fn,
    *,
    target_tokens: int = COMPRESS_REMOVE_TOKENS,
    min_n: int = COMPRESS_RANGE_MIN,
    max_n: int = COMPRESS_RANGE_MAX,
) -> int:
    """Pick how many oldest thinks to compress so cumulative tokens hit target.

    Returns the smallest N in [min_n, max_n] such that the sum of the
    first-N thinks' tokens >= target_tokens. If can't reach the target
    within max_n thinks, returns min(len(thinks), max_n). Returns 0 when
    len(thinks) < min_n (caller should not invoke compression in that state).

    Aligns inference-time range selection with pass2's range scoring,
    which was already implicitly token-driven via the hysteresis budget.
    """
    if len(thinks) < min_n:
        return 0
    cap = min(len(thinks), max_n)
    cum = 0
    for i in range(cap):
        cum += token_count_fn(thinks[i])
        if i + 1 >= min_n and cum >= target_tokens:
            return i + 1
    return cap


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _norm_for_memory_similarity(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _think_similarity(a: str, b: str) -> float:
    left = _norm_for_memory_similarity(a)
    right = _norm_for_memory_similarity(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _contiguous_groups(chunks: List[int]) -> List[List[int]]:
    if not chunks:
        return []
    uniq = sorted(set(int(c) for c in chunks))
    groups: List[List[int]] = [[uniq[0]]]
    for c in uniq[1:]:
        if c == groups[-1][-1] + 1:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def _memory_item_chunks(item: Dict) -> List[int]:
    raw_chunks = item.get("chunks")
    if raw_chunks is None:
        raw_chunks = item.get("source_chunks")
    if raw_chunks is not None:
        out: List[int] = []
        for c in raw_chunks:
            try:
                out.append(int(c))
            except (TypeError, ValueError):
                continue
        if out:
            return sorted(set(out))
    try:
        return [int(item["chunk"])]
    except (KeyError, TypeError, ValueError):
        return []


def _set_memory_item_chunks(item: Dict, chunks: List[int]) -> Dict:
    chunks = sorted(set(int(c) for c in chunks))
    if not chunks:
        return item
    item["chunk"] = chunks[0]
    item["chunks"] = chunks
    start = chunks[0] * AGENT_CHUNK_SEC
    end = (chunks[-1] + 1) * AGENT_CHUNK_SEC
    item["time"] = f"{int(start)}-{int(end)}"
    item["time_range"] = [int(start), int(end)]
    if len(chunks) > 1:
        item["range_merged"] = True
    else:
        item.pop("range_merged", None)
    return item


# ---------------------------------------------------------------------------
# Memory State (mirrors pass2_rollout.py:MemoryState)
# ---------------------------------------------------------------------------


class MemoryState:
    """Tracks the agent's text memory at each timestep.

    Mirrors scripts/agent_data_v5/pass2_rollout.py:MemoryState exactly.
    Text memory covers LONGER time than visual window.
    """

    def __init__(
        self,
        tokenizer=None,
        *,
        merge_similar_thinks: Optional[bool] = None,
        merge_similarity_threshold: Optional[float] = None,
    ):
        self.compressed_segments: List[Dict] = []
        self.recent_thinks: List[Dict] = []
        self._retrieval_archive: List[Dict] = []
        self._tokenizer = tokenizer
        self.merge_similar_thinks = (
            _env_bool("THINKSTREAM_MEMORY_MERGE_SIMILAR_THINKS", False)
            if merge_similar_thinks is None
            else bool(merge_similar_thinks)
        )
        self.merge_similarity_threshold = (
            _env_float("THINKSTREAM_MEMORY_MERGE_SIM_THRESHOLD", 0.96)
            if merge_similarity_threshold is None
            else float(merge_similarity_threshold)
        )

    @property
    def retrieval_archive(self) -> List[Dict]:
        return self._retrieval_archive

    def snapshot(self, chunk_idx: int) -> Dict:
        """Snapshot of what the model sees (no archive).

        Note: `pending_questions` was removed from this snapshot in
        v11.1 — it was always empty across all 12,405 v9.2 SFT samples
        (including the 184 recall_response variants). Pending question
        state is now expressed solely through the queries log
        (an entry with empty `answers` list = pending).
        format_memory_block in agent_protocol.py still tolerates the
        legacy field via .get() so older snapshots stay readable.
        """
        return {
            "chunk_idx": chunk_idx,
            "compressed_segments": deepcopy(self.compressed_segments),
            "recent_thinks": deepcopy(self.recent_thinks),
            "visual_window_start": max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1),
        }

    def add_think(self, chunk_idx: int, think_text: str) -> Optional[Dict]:
        """Add think to memory immediately.

        Returns a diagnostic event when the new think is merged into the
        previous visible memory item. Raw per-chunk archive entries are still
        appended before any visible-memory merge.
        """
        time_start = chunk_idx * AGENT_CHUNK_SEC
        time_end = time_start + AGENT_CHUNK_SEC
        item = {
            "chunk": chunk_idx,
            "chunks": [chunk_idx],
            "time": f"{int(time_start)}-{int(time_end)}",
            "time_range": [int(time_start), int(time_end)],
            "text": think_text,
        }
        self._retrieval_archive.append(dict(item))
        if self.merge_similar_thinks and self.recent_thinks:
            prev = self.recent_thinks[-1]
            prev_chunks = _memory_item_chunks(prev)
            if prev_chunks and max(prev_chunks) + 1 == int(chunk_idx):
                prev_text = str(prev.get("text", ""))
                prev_chunks_before = list(prev_chunks)
                prev_range_before = [
                    int(prev_chunks_before[0] * AGENT_CHUNK_SEC),
                    int((prev_chunks_before[-1] + 1) * AGENT_CHUNK_SEC),
                ]
                similarity = _think_similarity(prev_text, think_text)
                if similarity >= self.merge_similarity_threshold:
                    merged = _memory_item_chunks(prev) + [int(chunk_idx)]
                    _set_memory_item_chunks(prev, merged)
                    prev["merged_similar_count"] = (
                        int(prev.get("merged_similar_count", 1) or 1) + 1
                    )
                    prev["last_merge_similarity"] = round(float(similarity), 6)
                    merged_range = [
                        int(merged[0] * AGENT_CHUNK_SEC),
                        int((merged[-1] + 1) * AGENT_CHUNK_SEC),
                    ]
                    return {
                        "merged": True,
                        "reason": "adjacent_similarity",
                        "reason_detail": (
                            "previous visible memory_think is contiguous and "
                            "text similarity is above threshold"
                        ),
                        "similarity": round(float(similarity), 6),
                        "threshold": round(float(self.merge_similarity_threshold), 6),
                        "chunk_idx": int(chunk_idx),
                        "chunk_time_range": [int(time_start), int(time_end)],
                        "original_think": think_text,
                        "previous_text": prev_text,
                        "previous_chunks_before": prev_chunks_before,
                        "previous_time_range_before": prev_range_before,
                        "merged_chunks_after": sorted(set(int(c) for c in merged)),
                        "merged_time_range_after": merged_range,
                        "merged_duration_sec_after": int(merged_range[1] - merged_range[0]),
                        "merged_chunk_count_after": len(set(int(c) for c in merged)),
                    }
        self.recent_thinks.append(item)
        return None

    def chunks_for_items(self, items: List[Dict]) -> List[int]:
        chunks: List[int] = []
        for item in items:
            chunks.extend(_memory_item_chunks(item))
        return sorted(set(chunks))

    def chunks_in_time_range(self, time_range) -> List[int]:
        tr_start = tr_end = None
        if isinstance(time_range, list) and len(time_range) == 2:
            try:
                tr_start, tr_end = int(time_range[0]), int(time_range[1])
            except (TypeError, ValueError):
                tr_start = tr_end = None
        if tr_start is None or tr_end is None:
            return []
        selected: List[int] = []
        for item in self.recent_thinks:
            for c in _memory_item_chunks(item):
                chunk_start = int(c * AGENT_CHUNK_SEC)
                chunk_end = int(chunk_start + AGENT_CHUNK_SEC)
                if tr_start <= chunk_start and chunk_end <= tr_end:
                    selected.append(int(c))
        return sorted(set(selected))

    def _token_count(self, item: Dict) -> int:
        """Count tokens in a single recent_think entry."""
        text = item.get("text", "")
        if self._tokenizer:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return len(text) // 4

    def count_recent_tokens(self) -> int:
        """Count total tokens in recent_thinks."""
        return sum(self._token_count(item) for item in self.recent_thinks)

    def should_compress(self) -> bool:
        """Trigger compression when recent_thinks reach 80% of token budget."""
        return (
            self.count_recent_tokens() >= COMPRESS_TOKEN_THRESHOLD
            and len(self.recent_thinks) >= COMPRESS_RANGE_MIN
        )

    def compress(self, summary: Dict, compressed_chunks: Optional[List[int]] = None):
        """Replace specified thinks with summary in model context.

        Raw thinks stay in _retrieval_archive for recall.

        Cap = SFT SUMMARY_TOKENS_MAX (v11.3: 280 tok). Caps both incoming
        summary text and merged-segment text. Going above the cap is OOD
        relative to the SFT distribution.

        v11.3: When `compressed_chunks` is None (legacy fallback path),
        select the range via select_compress_range_by_tokens so post-
        compress memory drops by COMPRESS_REMOVE_TOKENS — aligns with
        pass2 hysteresis instead of always cutting exactly 4 thinks.
        """
        tr = summary.get("time_range") or []
        tr_start = tr_end = None
        if isinstance(tr, list) and len(tr) == 2:
            try:
                tr_start, tr_end = int(tr[0]), int(tr[1])
            except (TypeError, ValueError):
                tr_start = tr_end = None

        source_chunks = set()
        replaced_merge_levels = []
        if tr_start is not None and tr_end is not None:
            kept_segments = []
            for seg in self.compressed_segments:
                s_tr = seg.get("time_range") or []
                try:
                    s_start, s_end = [int(x) for x in s_tr[:2]]
                except (TypeError, ValueError):
                    s_start, s_end = (None, None)
                covered = s_start is not None and tr_start <= s_start and s_end <= tr_end
                if covered:
                    replaced_merge_levels.append(int(seg.get("merge_level", 0) or 0))
                    source_chunks.update(int(c) for c in seg.get("source_chunks", []) or [])
                    if not seg.get("source_chunks"):
                        source_chunks.update(range(s_start, s_end))
                else:
                    kept_segments.append(seg)
            self.compressed_segments = kept_segments

        chunk_set = set(int(c) for c in (compressed_chunks or []))
        if tr_start is not None and tr_end is not None:
            chunk_set.update(self.chunks_in_time_range([tr_start, tr_end]))

        if chunk_set:
            source_chunks.update(chunk_set)
            kept_recent: List[Dict] = []
            for t in self.recent_thinks:
                chunks = _memory_item_chunks(t)
                remaining = [c for c in chunks if int(c) not in chunk_set]
                if len(remaining) == len(chunks):
                    kept_recent.append(t)
                    continue
                for group in _contiguous_groups(remaining):
                    kept_recent.append(_set_memory_item_chunks(dict(t), group))
            self.recent_thinks = kept_recent
        else:
            n = select_compress_range_by_tokens(
                self.recent_thinks,
                token_count_fn=self._token_count,
            )
            source_chunks.update(self.chunks_for_items(self.recent_thinks[:n]))
            self.recent_thinks = self.recent_thinks[n:] if n > 0 else self.recent_thinks
        if self._tokenizer and isinstance(summary.get("text"), str):
            ids = self._tokenizer.encode(summary["text"], add_special_tokens=False)
            if len(ids) > SUMMARY_TOKENS_MAX:
                summary = dict(summary)
                summary["text"] = self._tokenizer.decode(ids[:SUMMARY_TOKENS_MAX])
                summary["_truncated"] = True
        summary = dict(summary)
        if source_chunks:
            summary["source_chunks"] = sorted(source_chunks)
        if "merge_level" not in summary:
            summary["merge_level"] = (
                max(replaced_merge_levels) + 1 if replaced_merge_levels else 1
            )
        self.compressed_segments.append(summary)

    # --- Queries tracking (matches SFT active_query/response_history zones) ---
    # The legacy add_pending / resolve_pending pair was removed in v11.1
    # — training data had pending_questions empty across all 12,405
    # samples, so the field was reverse-OOD at inference. "Pending" is
    # now expressed by add_query() leaving `answers=[]`; once
    # answer_query() runs, the entry becomes answered. format_memory_block
    # still tolerates the legacy field via .get() for back-compat with
    # any external snapshot dumps.

    def add_query(self, question: str, ask_time: float,
                   options: Optional[List[str]] = None,
                   answer_form: Optional[str] = None,
                   answer_style: Optional[str] = None,
                   answer_instruction: Optional[str] = None,
                   answer_chunks: Optional[List[int]] = None,
                   per_emit_answers: Optional[List[Dict]] = None,
                   open_until: Optional[float] = None):
        """Register a question (pending until answered).

        v12.13 fix (P0-1): accept options + answer_form so format_queries_block
        can render "Options: A) ... B) ..." for pending MC queries at
        inference / RL rollout time. v12.25 also stores answer_style /
        answer_instruction so SFT, RL, and eval see the same MC answer
        protocol hint. v12.43 carries answer_chunks/per_emit_answers so
        multi-emit questions remain active until their final expected response.
        """
        if not hasattr(self, "_queries"):
            self._queries = []
        expected_chunks = list(answer_chunks or [])
        expected_emits = list(per_emit_answers or [])
        answer_instruction = (
            canonical_answer_instruction({
                "answer_form": answer_form or "",
                "answer_style": answer_style or "",
                "answer_instruction": answer_instruction or "",
                "options": list(options or []),
            })
            or answer_instruction
            or ""
        )
        for q in reversed(self._queries):
            status = str(q.get("status", "")).strip().lower()
            if q.get("question") == question and status in {"open", "pending", "active"}:
                q["last_ask_time"] = ask_time
                if options:
                    q["options"] = list(options)
                if answer_form:
                    q["answer_form"] = answer_form
                if answer_style:
                    q["answer_style"] = answer_style
                if answer_instruction:
                    q["answer_instruction"] = answer_instruction
                if expected_chunks:
                    q["answer_chunks"] = expected_chunks
                if expected_emits:
                    q["per_emit_answers"] = expected_emits
                if open_until is None and expected_chunks:
                    try:
                        open_until = max(int(x) for x in expected_chunks) * AGENT_CHUNK_SEC
                    except (TypeError, ValueError):
                        open_until = None
                if open_until is not None:
                    q["open_until"] = open_until
                return
        for q in self._queries:
            status = str(q.get("status", "")).strip().lower()
            if status in {"open", "pending", "active"}:
                q["status"] = "replaced"
                q["closed_at"] = ask_time
                q["close_reason"] = "new_query"
        if open_until is None and expected_chunks:
            try:
                open_until = max(int(x) for x in expected_chunks) * AGENT_CHUNK_SEC
            except (TypeError, ValueError):
                open_until = None
        self._queries.append({
            "question": question,
            "ask_time": ask_time,
            "options": list(options or []),
            "answer_form": answer_form or "",
            "answer_style": answer_style or "",
            "answer_instruction": answer_instruction or "",
            "answer_chunks": expected_chunks,
            "per_emit_answers": expected_emits,
            "open_until": open_until,
            "status": "open",
            "answers": [],
        })

    def answer_query(
        self,
        question: str,
        answer: str,
        response_time: float,
        *,
        status: Optional[str] = None,
    ):
        """Record an answer for a pending query."""
        if not hasattr(self, "_queries"):
            return
        for q in reversed(self._queries):
            if q["question"] == question:
                append_query_answer_with_timing(q, answer, response_time)
                if status is not None:
                    q["status"] = status
                else:
                    q["status"] = "answered" if query_is_complete(q) else "open"
                return

    @property
    def queries(self) -> List[Dict]:
        return getattr(self, "_queries", [])


# format_memory_block, build_user_content are imported from
# thinkstream.data.agent_protocol (single source of truth). The v12 parser
# is wrapped by the local `_parse_agent_output` adapter above so the
# surrounding orchestration code keeps using the {action, payload} shape.


def build_single_step_messages(
    snapshot: Dict,
    chunk_idx: int,
    video_path: str,
    *,
    user_input: str = "",
    queries: Optional[List[Dict]] = None,
    recalled_frames: Optional[Dict] = None,
    recall_result: Optional[Dict] = None,
    # v12.12 (2026-05-02): RUNTIME profile defaults — see config.py
    min_pixels: int = 130_000,
    max_pixels: int = 220_000,
    frame_paths: Optional[List[str]] = None,
    frame_protocol: Optional[str] = None,
    inter_chunk: bool = False,
    render_layout: Optional[str] = None,
) -> List[Dict]:
    """Build single-step chat messages matching training format.

    Delegates text formatting to shared agent_protocol.build_user_content.
    Uses the protocol-aligned system prompt. The ``<tools>`` block is rendered
    by the chat_template from the turn-local schema returned by tools_for_turn().

    inter_chunk=True marks a memory-compaction turn. It suppresses query /
    recalled-answer context and the visual sliding window.
    """
    layout = normalize_render_layout(render_layout)
    memory_text = format_memory_block(snapshot)
    post_recall = bool(recall_result or recalled_frames) and not inter_chunk
    if post_recall:
        # A post-recall turn has no new current chunk. It contains only the
        # historical visual evidence returned by recall plus routing metadata.
        user_content = build_recall_result_user_content(
            recalled_frames,
            recall_result,
            frame_protocol=frame_protocol,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            render_layout=layout,
        )
    else:
        user_content = build_user_content(
            memory_text,
            chunk_idx,
            video_path,
            user_input=user_input,
            queries=queries,
            recalled_frames=None,
            recall_result=None,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            frame_paths=frame_paths,
            frame_protocol=frame_protocol,
            inter_chunk=inter_chunk,
            memory_snapshot=snapshot,
            render_layout=layout,
        )

    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": system_prompt_for_frame_protocol(
                    frame_protocol,
                    prompt_kind=("post_recall" if post_recall else None),
                    inter_chunk=inter_chunk,
                    render_layout=layout,
                ),
            }],
        },
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Simple Retrieval (for recall action)
# ---------------------------------------------------------------------------


def parse_time_range(tr) -> Optional[tuple]:
    """Parse a query['time_range'] field into (t_start, t_end) seconds.

    Accepts: "10-30", "10.0-30.0", [10, 30], (10, 30). Returns None on
    missing/empty/malformed input — callers should treat None as "no
    range filter; use full archive".
    """
    if tr is None:
        return None
    if isinstance(tr, (list, tuple)) and len(tr) == 2:
        try:
            return float(tr[0]), float(tr[1])
        except (TypeError, ValueError):
            return None
    if isinstance(tr, str):
        s = tr.strip()
        if not s:
            return None
        try:
            a, b = s.split("-", 1)
            return float(a), float(b)
        except (ValueError, AttributeError):
            return None
    return None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def recall_time_range_margin_chunks() -> int:
    """Small recall-range tolerance for RL exploration.

    The model still emits the exact requested range, but retrieval widens that
    range by a few chunks before scoring. This gives near-boundary recall
    attempts a chance to retrieve the support evidence and receive downstream
    answer reward. Set THINKSTREAM_RECALL_TIME_RANGE_MARGIN_CHUNKS=0 to recover
    strict historical behavior.
    """
    return max(0, _env_int("THINKSTREAM_RECALL_TIME_RANGE_MARGIN_CHUNKS", 3))


def expand_time_range_by_chunks(
    time_range,
    *,
    margin_chunks: int,
    chunk_sec: float = AGENT_CHUNK_SEC,
) -> Optional[tuple]:
    tr = parse_time_range(time_range)
    if tr is None:
        return None
    t0, t1 = tr
    if t0 > t1:
        t0, t1 = t1, t0
    margin = max(0, int(margin_chunks)) * float(chunk_sec)
    if margin <= 0:
        return t0, t1
    return max(0.0, t0 - margin), t1 + margin


def filter_archive_by_time_range(
    archive: List[Dict],
    time_range,
    chunk_sec: float = AGENT_CHUNK_SEC,
    *,
    margin_chunks: Optional[int] = None,
) -> List[Dict]:
    """Restrict archive to items whose chunk overlaps [t_start, t_end].

    "with_time_range" mode = the model emits a time_range and the retriever
    pre-filters to that window before scoring. A small margin can be applied
    around valid ranges so near-boundary recalls still retrieve useful support
    during RL exploration. Falls back to the full archive when the range is
    missing/malformed (matches the SFT distribution where ~30% of queries are
    keyword-only by design).
    """
    if margin_chunks is None:
        margin_chunks = recall_time_range_margin_chunks()
    tr = expand_time_range_by_chunks(
        time_range,
        margin_chunks=margin_chunks,
        chunk_sec=chunk_sec,
    )
    if tr is None:
        return archive
    t0, t1 = tr
    if t0 > t1:
        t0, t1 = t1, t0
    out = []
    for item in archive:
        c = item.get("chunk")
        if c is None:
            continue
        c_start = c * chunk_sec
        c_end = c_start + chunk_sec
        if c_end > t0 and c_start < t1:
            out.append(item)
    return out


def bm25_retrieve(
    query: Dict,
    archive: List[Dict],
    max_results: int = RECALL_RETURN_CHUNKS,
) -> Dict:
    """BM25-based retrieval from archive.

    Honours `query["time_range"]` when present (filters archive to chunks
    overlapping that window); falls back to full archive on missing /
    malformed range. Uses rank_bm25 if available, else keyword overlap.
    Returns an internal retrieval dict with text_content and returned_chunks.
    Runtime prompt rendering strips text_content and exposes metadata only.
    """
    query_text = query.get("query", "")
    if not query_text.strip() or not archive:
        return {
            "source": "failure",
            "time": "",
            "text_content": "No matching results found.",
            "returned_chunks": [],
        }

    margin_chunks = recall_time_range_margin_chunks()
    archive = filter_archive_by_time_range(
        archive,
        query.get("time_range"),
        margin_chunks=margin_chunks,
    )
    if not archive:
        return {
            "source": "failure",
            "time": "",
            "text_content": "No matching results found.",
            "returned_chunks": [],
        }

    texts = [item.get("text", "") for item in archive]

    try:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query_text.lower().split())
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:max_results]
    except ImportError:
        # Fallback: keyword overlap scoring
        query_words = set(query_text.lower().split())
        scored = []
        for i, text in enumerate(texts):
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            scored.append((overlap, i))
        scored.sort(key=lambda x: -x[0])
        top_indices = [i for _, i in scored[:max_results]]

    if not top_indices:
        return {
            "source": "failure",
            "time": "",
            "text_content": "No matching results found.",
            "returned_chunks": [],
        }

    top_items = [archive[i] for i in top_indices]
    returned_chunks = select_recall_chunks(
        [item["chunk"] for item in top_items],
        max_chunks=max_results,
    )
    returned_set = set(returned_chunks)
    text_parts = [
        f'[{item["time"]}] {item["text"]}'
        for item in top_items
        if int(item.get("chunk", -1)) in returned_set
    ]

    return {
        "source": "historical_frames",
        "time": recall_time_string_for_chunks(returned_chunks),
        "text_content": "\n".join(text_parts),
        "returned_chunks": returned_chunks,
        "query_time_range": query.get("time_range"),
        "time_range_margin_chunks": margin_chunks,
    }


# Backward compat alias
simple_retrieve = bm25_retrieve


# ---------------------------------------------------------------------------
# Generate Function Adapter
# ---------------------------------------------------------------------------


def make_generate_fn(
    model,
    processor,
    model_type: str = "qwen3vl",
    device: str = "cuda",
):
    """Create a generate_fn compatible with StreamingAgentLoop.

    Wraps a HuggingFace model (Qwen2.5-VL / Qwen3-VL) into a callable:
        generate_fn(messages, processor, max_new_tokens, **kwargs) -> str

    This uses standard HF generate (no CUDA graph / StreamingInferenceEngine).
    For production, replace with vLLM or StreamingInferenceEngine adapter.
    """
    import torch
    from thinkstream.data.stream_data_processor import compute_position_ids

    @torch.inference_mode()
    def generate_fn(
        messages,
        processor,
        max_new_tokens=256,
        **kwargs,
    ) -> str:
        # 1. Apply chat template + process vision (tokenize=True handles images/videos)
        # v12.15: pass the turn-local tool schema. Streaming turns expose
        # recall only, compression turns expose compress only, and recall-result
        # answer turns pass no tools.
        from thinkstream.data.agent_protocol import tools_for_turn
        # v12.6: collect per-video metadata for correct timestamp rendering
        video_metadata = []
        has_video_meta = True
        for msg in messages:
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "video":
                    meta = item.get("video_metadata")
                    frames = item.get("video")
                    if isinstance(meta, dict):
                        video_metadata.append({
                            k: v for k, v in meta.items()
                            if k != "do_sample_frames"
                        })
                    elif isinstance(frames, list) and frames:
                        # auto-fill so VideoMetadata() doesn't crash on empty dict
                        from thinkstream.data.agent_protocol import infer_video_metadata
                        video_metadata.append(infer_video_metadata(frames))
                    else:
                        has_video_meta = False
        template_kwargs = dict(
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_sample_frames=False,
        )
        tools = kwargs.get("tools")
        if tools is None and "tool_turn_kind" in kwargs:
            tools = tools_for_turn(kwargs.get("tool_turn_kind"))
        if tools is not None:
            template_kwargs["tools"] = tools
        if video_metadata and has_video_meta:
            template_kwargs["video_metadata"] = video_metadata
        inputs = processor.apply_chat_template(messages, **template_kwargs)

        # 2. Move to device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 3. Generate. Accuracy eval should be deterministic by default;
        # callers can opt back into sampling with do_sample=True or
        # temperature>0 for exploratory demos.
        temperature = float(kwargs.get("temperature", 0.0))
        do_sample = bool(kwargs.get("do_sample", temperature > 0.0))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_k": kwargs.get("top_k", 50),
                "top_p": kwargs.get("top_p", 0.95),
            })
        output_ids = model.generate(**inputs, **gen_kwargs)

        # 4. Decode (only new tokens)
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, input_len:]
        return processor.tokenizer.decode(new_ids, skip_special_tokens=False)

    return generate_fn


# ---------------------------------------------------------------------------
# Streaming Agent Loop
# ---------------------------------------------------------------------------


class StreamingAgentLoop:
    """Single-step inference loop matching training data format exactly.

    Each step constructs a complete single-turn input (memory + visual_window
    + user_input) and runs a fresh forward pass. No KV cache reuse across steps.

    Usage::

        loop = StreamingAgentLoop(generate_fn, tokenizer, processor)
        for chunk_idx in range(num_chunks):
            result = loop.step(
                chunk_idx=chunk_idx,
                video_path=video_path,
                user_question=question if chunk_idx == ask_chunk else None,
            )
            if result["action"] == "response":
                print(result["payload"]["response"])
                break
    """

    def __init__(
        self,
        generate_fn: Callable,
        tokenizer,
        processor,
        *,
        model_type: str = "qwen3vl",
        # v12.12 (2026-05-02): RUNTIME profile defaults
        min_pixels: int = 130_000,
        max_pixels: int = 220_000,
        max_new_tokens: int = 256,
        retrieve_fn: Optional[Callable] = None,
        retriever=None,
        compress_mode: str = "system",
        memory_mode: Optional[str] = None,
        frames_root: Optional[str] = None,
        video_root: Optional[str] = None,
        frame_protocol: Optional[str] = None,
    ):
        """
        Args:
            generate_fn: Callable that takes (messages, processor, **kwargs)
                         and returns generated text string.
            tokenizer: Tokenizer for token counting.
            processor: HuggingFace processor for tokenization + vision.
            model_type: "qwen2.5vl" or "qwen3vl".
            retrieve_fn: Optional custom retrieval function (legacy
                         interface). Use `retriever` instead for new code.
                         Kept for backward compat with callers that pass a
                         plain (query, archive) -> dict callable.
            retriever:   Optional Retriever instance (BM25Retriever or
                         HybridRetriever from thinkstream.model.retrieval).
                         Takes precedence over retrieve_fn. Stateful — its
                         index_chunk() is called after each chunk's think
                         is added so dense backends can build a visual index
                         on the fly.
            compress_mode: "system" (default, used by SFT eval) — when
                memory.should_compress() fires, system inserts a bare
                <compress_trigger/> as a memory-pressure signal (NO range,
                v12.12); the model derives the range from <memory> and
                writes both range and summary in its tool_call.
                "self" (used by RL eval after GDPO) — system never
                inserts a trigger; the model decides autonomously when
                to emit <action>compress</action> and which range to
                summarize. Only enable "self" with an RL-tuned ckpt:
                v11 SFT samples were all C1 (system-triggered fixed
                range), so a pure-SFT model under "self" mode is OOD.
            memory_mode: eval ablation switch. "full" keeps text memory in
                ordinary prompts and recall archives. "no_prompt" hides text
                memory from ordinary prompts but still records it for
                recall/compression. "no_recall" keeps prompt memory but makes
                recall retrieve nothing. "none" disables text memory prompt,
                recall archive, and compression.
        """
        compress_mode = str(compress_mode or "system").strip().lower()
        if compress_mode == "none":
            compress_mode = "off"
        if compress_mode not in ("system", "self", "off"):
            raise ValueError(
                "compress_mode must be 'system', 'self', or 'off', "
                f"got {compress_mode!r}"
            )
        memory_mode = str(
            memory_mode
            or os.environ.get("THINKSTREAM_EVAL_MEMORY_MODE", "full")
            or "full"
        ).strip().lower().replace("-", "_")
        memory_mode = {
            "prompt_off": "no_prompt",
            "no_memory_prompt": "no_prompt",
            "disable_prompt": "no_prompt",
            "recall_off": "no_recall",
            "disable_recall": "no_recall",
            "off": "none",
            "no_memory": "none",
        }.get(memory_mode, memory_mode)
        if memory_mode not in {"full", "no_prompt", "no_recall", "none"}:
            raise ValueError(
                "memory_mode must be full, no_prompt, no_recall, or none, "
                f"got {memory_mode!r}"
            )
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer       # needed for telemetry token counts
        self.processor = processor
        self.model_type = model_type
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.frame_protocol = normalize_frame_protocol(frame_protocol)
        self.render_layout = normalize_render_layout()
        # Resolve retriever: explicit `retriever` > `retrieve_fn` > BM25 default.
        # The new Retriever API has both __call__ and index_chunk; legacy
        # retrieve_fn callables are wrapped via coerce_retriever.
        from thinkstream.model.retrieval import coerce_retriever, BM25Retriever
        if retriever is not None:
            self.retriever = coerce_retriever(retriever)
        elif retrieve_fn is not None:
            self.retriever = coerce_retriever(retrieve_fn)
        else:
            self.retriever = BM25Retriever()
        # retrieve_fn kept as a thin alias for legacy access.
        self.retrieve_fn = self.retriever
        self.compress_mode = compress_mode
        self.memory_mode = memory_mode
        self.frames_root = frames_root
        self.video_root = video_root
        self.memory = MemoryState(tokenizer=tokenizer)
        # v12.6: pre-init the captured-prompt buffer so early-exit paths in
        # step() (raised exception before line 797 assignment) don't leave
        # downstream readers (parsed["step_messages"] = self._last_step_messages
        # near line 1067) with an AttributeError. None is a safe sentinel —
        # _build_rollout_messages._captured_for_gen() falls back to legacy
        # reconstruction when the field is None.
        self._last_step_messages: Optional[List[Dict]] = None

    def _ordinary_prompt_snapshot(self, snapshot: Dict) -> Dict:
        """Apply text-memory ablation to ordinary streaming prompts."""
        if self.memory_mode not in {"no_prompt", "none"}:
            return snapshot
        out = dict(snapshot)
        out["compressed_segments"] = []
        out["compressed"] = []
        out["recent_thinks"] = []
        return out

    def _get_frame_paths(self, video_path: str, chunk_idx: int) -> Optional[List[str]]:
        """Build frame_paths for the current visual_window from pre-extracted frames."""
        if not self.frames_root:
            return None
        window_start = max(0, chunk_idx - VISUAL_WINDOW_CHUNKS + 1)
        video_start = window_start * AGENT_CHUNK_SEC
        video_end = (chunk_idx + 1) * AGENT_CHUNK_SEC
        n_frames = (chunk_idx - window_start + 1) * FRAMES_PER_CHUNK

        vp = Path(video_path)
        # Try relative path under video_root, fallback to full relative path
        if self.video_root:
            try:
                rel = vp.relative_to(Path(self.video_root))
                stem = rel.with_suffix("")
                frame_dir = Path(self.frames_root) / stem
            except ValueError:
                frame_dir = Path(self.frames_root) / vp.with_suffix("")
        else:
            frame_dir = Path(self.frames_root) / vp.with_suffix("")

        # v12.6 fix (2026-04-30): videos may come from multiple source datasets
        # with different directory structures. The nested path above only works
        # when video_root matches; for cross-dataset backups or mixed pools,
        # frames are stored flat as {frames_root}/{video_stem}. Try flat lookup
        # as a fallback before giving up.
        if not frame_dir.exists():
            flat_dir = Path(self.frames_root) / vp.stem
            if flat_dir.exists():
                frame_dir = flat_dir
            else:
                return None

        # v12.6 fix: index frames by chunk_idx × FRAMES_PER_CHUNK + 1.
        # The seconds-based variant (int(video_start)+1) was off-by-half
        # under FPS=2 — matches pass1a_evidence.get_chunk_frame_paths so
        # SFT and inference read the same frame set per chunk.
        frame_paths = []
        for ci in range(window_start, chunk_idx + 1):
            for fi in range(FRAMES_PER_CHUNK):
                fnum = ci * FRAMES_PER_CHUNK + fi + 1
                fp = frame_dir / f"frame_{fnum:06d}.jpg"
                if fp.exists():
                    frame_paths.append(str(fp))

        # If too few frames found, fall back to online decoding
        if len(frame_paths) < max(1, n_frames // 2):
            return None
        return frame_paths

    def reset(self):
        """Reset for a new video."""
        self.memory = MemoryState(tokenizer=self.memory._tokenizer)
        # v12.6: clear stale captured prompt from previous video.
        self._last_step_messages = None

    def _record_answer(self, answer_text: str, chunk_idx: int) -> None:
        """Attach an answer to the most-recent unanswered query.

        Mirrors how pass3c builds queries_state when the agent produces
        a response: walk queries in reverse, find the first one with
        empty answers, append. If no unanswered query exists (e.g. the
        model emitted a stray response without a pending question),
        silently no-op rather than fabricating a Q to attach to.
        """
        if not answer_text:
            return
        response_time = chunk_idx * AGENT_CHUNK_SEC
        for q in reversed(self.memory.queries):
            status = str(q.get("status", "")).strip().lower()
            if status in {"open", "pending", "active"} or (
                not status and not q.get("answers")
            ):
                self.memory.answer_query(q["question"], answer_text, response_time)
                return

    def step(
        self,
        chunk_idx: int,
        video_path: str,
        user_question: Optional[str] = None,
        user_question_meta: Optional[Dict] = None,
        **generate_kwargs,
    ) -> Dict:
        """Execute one agent step (one chunk).

        Returns parsed output dict with keys: think, action, payload.
        Handles compression trigger and recall orchestration internally.

        v12.13 fix (P0-1): user_question_meta carries options + answer_form
        for MC queries so MemoryState.add_query stores them; subsequent
        chunks render Options in the active-query block via format_queries_block.
        """
        # 1. Snapshot BEFORE this step. Compression turns must still see the
        # full memory state; ordinary turns may hide text memory for ablations.
        full_snapshot = self.memory.snapshot(chunk_idx)

        # 1b. Register the new question (if any) into the queries log so
        # it appears in the active-query block this step. Training data has
        # the question present in query state at the chunk it arrives —
        # 7,828/12,405 v9.2 samples (63%) carry populated queries — so
        # not registering it would leave the model in an OOD distribution
        # for any chunk after the first question. Idempotent: same
        # (question, ask_time) is appended only once.
        if user_question:
            ask_time = chunk_idx * AGENT_CHUNK_SEC
            already_logged = any(
                q["question"] == user_question and q.get("ask_time") == ask_time
                for q in self.memory.queries
            )
            if not already_logged:
                meta = user_question_meta or {}
                self.memory.add_query(
                    user_question, ask_time,
                    options=meta.get("options"),
                    answer_form=meta.get("answer_form"),
                    answer_style=meta.get("answer_style"),
                    answer_instruction=meta.get("answer_instruction"),
                    answer_chunks=meta.get("answer_chunks"),
                    per_emit_answers=meta.get("per_emit_answers"),
                    open_until=meta.get("open_until"),
                )

        # 2. Check compression trigger (system-triggered, not model-triggered).
        #
        # v11.3: range size is token-driven via select_compress_range_by_tokens
        # (was hardcoded to COMPRESS_RANGE_MIN=4). Pass2 already enumerated
        # variable ranges in [4, 8] via score_range_for_compression; agent_loop
        # now matches that variability so inference and training agree on
        # the policy. v12.12+ injects only a bare <compress_trigger/>; the
        # model derives and emits time_range inside the compress tool_call.
        compress_trigger = ""
        # v9.4.2: telemetry for streaming eval — record state at the moment
        # compression FIRES so eval can stat: how many thinks were buffered
        # (vs the 480-tok / 4-think threshold) and which chunks got rolled
        # into the summary. Set on parsed below.
        _compress_telemetry = None
        if (
            self.compress_mode == "system"
            and self.memory_mode != "none"
            and self.memory.should_compress()
        ):
            n_to_compress = select_compress_range_by_tokens(
                self.memory.recent_thinks,
                token_count_fn=self.memory._token_count,
            )
            oldest = self.memory.recent_thinks[:n_to_compress] if n_to_compress > 0 else []
            if oldest:
                chunks = self.memory.chunks_for_items(oldest)
                # v12.12 (2026-05-02): trigger carries NO range. Model must
                # derive the range from <memory> contents and emit it inside
                # the assistant tool_call. Range comparison for telemetry /
                # success scoring (downstream) still uses `chunks` computed
                # by the system's range-selection policy as the oracle target.
                compress_trigger = "<compress_trigger/>"
                _compress_telemetry = {
                    "thinks_count_at_trigger": len(self.memory.recent_thinks),
                    "thinks_token_count": self.memory.count_recent_tokens(),
                    "compressed_chunks": chunks,
                    "trigger_chunk": chunk_idx,
                    "compress_threshold": COMPRESS_TOKEN_THRESHOLD,
                    "compress_range_min": COMPRESS_RANGE_MIN,
                    "compress_range_max": COMPRESS_RANGE_MAX,
                    "system_trigger_rule_ok": (
                        self.memory.count_recent_tokens() >= COMPRESS_TOKEN_THRESHOLD
                        and len(self.memory.recent_thinks) >= COMPRESS_RANGE_MIN
                    ),
                    "system_range_rule_ok": (
                        COMPRESS_RANGE_MIN <= len(chunks) <= COMPRESS_RANGE_MAX
                    ),
                }
        # compress_mode == "self": no trigger inserted. The model is
        # expected to autonomously emit <action>compress</action> when
        # it judges memory pressure, with its own time_range in the
        # <summary>. Only used after GDPO has trained the policy to
        # pick ranges; pure-SFT ckpts will likely never compress in
        # this mode and overflow.

        # 3. Determine user_input
        user_input = ""
        if compress_trigger:
            # Compression is a system memory-management turn. It preempts
            # visual/question turns and does not consume the current video
            # chunk; callers that maintain their own chunk cursor should retry
            # this chunk after a successful compression.
            user_input = compress_trigger
        elif user_question:
            user_input = user_question

        # 4. Build single-step messages (matching training format).
        # When compression fires, mark inter_chunk=True so the prompt uses the
        # compression-only system prompt and omits query/visual context.
        is_inter_chunk = bool(compress_trigger)
        snapshot = (
            full_snapshot
            if is_inter_chunk
            else self._ordinary_prompt_snapshot(full_snapshot)
        )
        frame_paths = self._get_frame_paths(video_path, chunk_idx)
        messages = build_single_step_messages(
            snapshot,
            chunk_idx,
            video_path,
            user_input=user_input,
            queries=self.memory.queries,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            frame_paths=frame_paths,
            frame_protocol=self.frame_protocol,
            inter_chunk=is_inter_chunk,
            render_layout=self.render_layout,
        )
        # v12.6: stash the EXACT messages used for generation so RL loss-time
        # reconstruction can replay the same prompt — see
        # thinkstream/trainer/grpo.py:_build_rollout_messages. Without this
        # the loss path conditions logprobs on a stripped-down context (no
            # <memory>, <visual_window>, active-query state) and gradient direction drifts.
        self._last_step_messages = messages

        # 5. Generate
        tool_turn_kind = "compress" if is_inter_chunk else "streaming"
        turn_max_new_tokens = 512 if tool_turn_kind == "compress" else self.max_new_tokens
        output_text = self.generate_fn(
            messages=messages,
            processor=self.processor,
            max_new_tokens=turn_max_new_tokens,
            tool_turn_kind=tool_turn_kind,
            tools=tools_for_turn(tool_turn_kind),
            **generate_kwargs,
        )

        # 6. Parse output
        parsed = _parse_agent_output(output_text)
        action_error = action_space_error_for_turn(
            parsed.get("action", ""),
            tool_turn_kind,
        )
        if action_error:
            parsed["action_space_error"] = action_error
            parsed["invalid_action"] = parsed.get("action", "")
            parsed["action"] = "invalid"

        if os.environ.get("AGENT_DEBUG"):
            print(f"[AGENT_DEBUG] chunk={chunk_idx} user_input={user_input!r}")
            print(f"[AGENT_DEBUG] raw_output={output_text!r}")
            print(f"[AGENT_DEBUG] parsed action={parsed['action']!r} think_len={len(parsed['think'])}")

        # 7. Update memory state based on action. Compress turns are
        # memory-management tool calls, not video observations, so their
        # <think> is not inserted into recent_thinks / recall archive.
        if (
            parsed["think"]
            and parsed["action"] != "compress"
            and not is_inter_chunk
            and self.memory_mode != "none"
        ):
            self.memory.add_think(chunk_idx, parsed["think"])
            # Stateful retrievers (e.g. HybridRetriever) hook here to
            # encode the chunk's frames into their visual index. BM25Retriever
            # no-ops. Failures are swallowed so retrieval doesn't break the
            # main agent loop.
            try:
                self.retriever.index_chunk(chunk_idx, video_path, parsed["think"])
            except Exception as e:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "retriever index_chunk failed (chunk=%d): %s", chunk_idx, e
                )

        if parsed["action"] == "compress":
            summary = parsed["payload"].get("summary", {})
            if summary and "time_range" in summary:
                # Determine which chunks were compressed from time_range
                compressed_chunks = self.memory.chunks_in_time_range(
                    summary["time_range"]
                )
                self.memory.compress(summary, compressed_chunks=compressed_chunks)

        elif parsed["action"] == "recall":
            # Orchestrate recall: retrieve → build recall_response input → second generate
            query = parsed["payload"].get("query", {})
            if query:
                recall_archive = (
                    []
                    if self.memory_mode in {"no_recall", "none"}
                    else self.memory.retrieval_archive
                )
                raw_recall_result = self.retriever(query, recall_archive)
                returned_chunks = select_recall_chunks(
                    raw_recall_result.get("returned_chunks", [])
                )
                raw_recall_result["returned_chunks"] = returned_chunks

                # Build recalled_frames info (including frame_paths so we
                # don't fallback to full-video decoding in recall_response).
                recalled_frames = None
                if returned_chunks and raw_recall_result.get("source") == "historical_frames":
                    rf_paths = []
                    frame_chunks = []
                    # Build recalled frame_paths by resolving per-chunk frames
                    # under the same frames_root logic.
                    if self.frames_root:
                        vp = Path(video_path)
                        if self.video_root:
                            try:
                                rel = vp.relative_to(Path(self.video_root))
                                stem = rel.with_suffix("")
                                frame_dir = Path(self.frames_root) / stem
                            except ValueError:
                                frame_dir = Path(self.frames_root) / vp.with_suffix("")
                        else:
                            frame_dir = Path(self.frames_root) / vp.with_suffix("")
                        if frame_dir.exists():
                            # v12.6 fix: same chunk×FRAMES_PER_CHUNK convention
                            # used everywhere else (pass1a, _get_frame_paths,
                            # streaming_vllm). Old code used seconds-based
                            # offsets which were off-by-half under FPS=2.
                            for rc in returned_chunks:
                                chunk_paths = []
                                for fi in range(FRAMES_PER_CHUNK):
                                    fnum = rc * FRAMES_PER_CHUNK + fi + 1
                                    fp = frame_dir / f"frame_{fnum:06d}.jpg"
                                    if fp.exists():
                                        chunk_paths.append(str(fp))
                                if chunk_paths:
                                    frame_chunks.append(rc)
                                    rf_paths.extend(chunk_paths)
                    recalled_frames = build_recalled_frames_metadata(
                        frame_chunks if rf_paths else returned_chunks,
                        rf_paths,
                        chunk_sec=AGENT_CHUNK_SEC,
                        frames_per_chunk=FRAMES_PER_CHUNK,
                    )
                recall_result = build_recall_result_metadata(
                    raw_recall_result,
                    recalled_frames,
                )

                # v12.6 fix: build true multi-turn recall prompt matching
                # SFT shape B (pass5_messages.py:212-260).
                # Old behavior REBUILT a fresh single-turn prompt with
                # recall_result inlined into the same user content, dropping
                # the model's own recall tool_call from context — train/infer
                # divergence. New behavior:
                #   [system, user(chunk N), assistant(recall tool_call),
                #    user(recall_result + recalled_frames)] → generate answer
                # This is byte-identical to the SFT trajectory the model saw.
                recall_messages = deepcopy(messages)         # [system, user(chunk N)]
                if recall_messages and recall_messages[0].get("role") == "system":
                    recall_messages[0] = {
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": system_prompt_for_frame_protocol(
                                self.frame_protocol,
                                prompt_kind="post_recall",
                                render_layout=self.render_layout,
                            ),
                        }],
                    }
                recall_messages.append({                      # model's own recall turn
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                })
                tool_user_content = build_recall_result_user_content(
                    recalled_frames,
                    recall_result,
                    frame_protocol=self.frame_protocol,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    render_layout=self.render_layout,
                )
                recall_messages.append({
                    "role": "user", "content": tool_user_content,
                })

                # Second generate (allow_recall=False to prevent infinite loop)
                recall_gen_kwargs = dict(generate_kwargs)
                recall_gen_kwargs["allow_recall"] = False
                recall_gen_kwargs["tool_turn_kind"] = "post_recall"
                recall_gen_kwargs["tools"] = tools_for_turn("post_recall")
                recall_output_text = self.generate_fn(
                    messages=recall_messages,
                    processor=self.processor,
                    max_new_tokens=self.max_new_tokens,
                    **recall_gen_kwargs,
                )

                recall_parsed = _parse_agent_output(recall_output_text)
                recall_action_error = action_space_error_for_turn(
                    recall_parsed.get("action", ""),
                    "post_recall",
                )
                # post_recall may contain a local <think> over the tool
                # result, but it is not a new video observation and is not
                # inserted into recent_thinks / retrieval archive. The
                # timestep memory was already emitted and indexed in turn 1.

                # v12.6: post-parse recall-budget enforcement.
                # The `allow_recall=False` flag became a no-op when v12.6
                # stripped restricted-decoding (think_budget_sample_*) from
                # inference.py. Replace it with a post-parse guard: if the
                # second pass still emits another tool_call (recall or
                # compress), override to "silent" and stash the offending
                # text under `recall_step2_blocked` for telemetry. Matches
                # DeepEyesV2 vl_agent.py recall_budget=1 contract; SFT shape B
                # already trains "answer after recall_result" but a guard is
                # cheap insurance against OOD drift / low-quality retrieval.
                if recall_action_error or recall_parsed["action"] in ("recall", "compress"):
                    parsed["recall_step2_blocked"] = {
                        "action": recall_parsed["action"],
                        "action_space_error": recall_action_error,
                        "raw_output": recall_output_text,
                    }
                    recall_parsed = {
                        "action": "silent",
                        "payload": {},
                        "raw_output": "",
                    }

                # Merge recall results into parsed output
                parsed["recall_step2"] = recall_parsed
                parsed["recall_step2_raw_text"] = recall_output_text  # v12.11 P0.6 — exposed for loss
                parsed["recall_result"] = recall_result
                # Override action to the final action (response or silent)
                if recall_parsed["action"] in ("response", "silent"):
                    parsed["final_action"] = recall_parsed["action"]
                    parsed["final_payload"] = recall_parsed["payload"]
                # If the recall second pass emitted a response, log the
                # answer against the most recent unanswered query so the
                # next chunk's response_history carries it forward if the
                # query is still active.
                if recall_parsed["action"] == "response":
                    answer_text = recall_parsed["payload"].get("response", "")
                    self._record_answer(answer_text, chunk_idx)

        elif parsed["action"] == "response":
            # Log the answer in the queries log so it shows up in the
            # next chunk's response_history if the query remains active. We
            # attribute it to the most
            # recent unanswered query, which matches how training data
            # was generated (pass3c emits Q/A pairs in arrival order)
            # and how an unanswered query implicitly represents pending
            # status (no separate pending_questions field needed).
            answer_text = parsed["payload"].get("response", "")
            self._record_answer(answer_text, chunk_idx)

        # Expose post-step memory size so RL rollouts can detect overflow
        # (recent_thinks tokens; compressed_segments are bounded by design).
        parsed["memory_token_count"] = self.memory.count_recent_tokens()
        parsed["compress_threshold"] = COMPRESS_TOKEN_THRESHOLD
        parsed["compress_budget"] = RECENT_THINKS_TOKEN_BUDGET
        # v9.4.2 telemetry: streaming eval reads these per-step and aggregates.
        # `compress_telemetry` is non-None ONLY when compression fired this step
        # (regardless of whether the model produced a valid <summary> response).
        # `recall_returned_chunks` is populated when an `action=recall` fired.
        parsed["compress_telemetry"] = _compress_telemetry
        if parsed.get("recall_result"):
            parsed["recall_returned_chunks"] = parsed["recall_result"].get(
                "returned_chunks", []
            )
        else:
            parsed["recall_returned_chunks"] = []

        # ── v9.4.2 extra telemetry (4 metrics) ──
        # 1. prompt_text_token_count: text-only zones (system + user_input +
        #    memory + queries + recall). Visual frames excluded — they're a fixed
        #    cost the eval can compute as 24 × ~196 = ~4700. Sum the two gives
        #    a per-step "how close are we to model_max_length" signal.
        prompt_text_tokens = 0
        if self.tokenizer is not None:
            try:
                # Concat all text-type content from messages (excludes video/image dicts).
                text_acc = []
                for msg in messages:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_acc.append(item.get("text", ""))
                    elif isinstance(content, str):
                        text_acc.append(content)
                if text_acc:
                    prompt_text_tokens = len(self.tokenizer.encode(
                        "\n".join(text_acc), add_special_tokens=False))
            except Exception:
                prompt_text_tokens = 0
        parsed["prompt_text_token_count"] = prompt_text_tokens

        # 2. think_token_count: chattiness probe — does the model's <think>
        #    grow over time? SFT trained at THINK_TOKENS [25, 130]; large
        #    values signal verbosity drift / format breakdown.
        think_tokens = 0
        if parsed.get("think") and self.tokenizer is not None:
            try:
                think_tokens = len(self.tokenizer.encode(
                    parsed["think"], add_special_tokens=False))
            except Exception:
                think_tokens = 0
        parsed["think_token_count"] = think_tokens

        # 3. format_ok: did the output have a parseable <think> AND <action>?
        #    Action-specific payload presence is also required for non-silent.
        VALID_ACTIONS = {"silent", "response", "recall", "compress"}
        action = parsed.get("action") or ""
        format_ok = (
            bool(parsed.get("think"))
            and action in VALID_ACTIONS
            and not parsed.get("action_space_error")
        )
        if format_ok:
            payload = parsed.get("payload") or {}
            if action == "response":
                format_ok = "response" in payload and bool(payload["response"])
            elif action == "recall":
                format_ok = "query" in payload  # parsed JSON; query_raw means JSON broke
            elif action == "compress":
                summary = payload.get("summary")
                format_ok = bool(summary) and "time_range" in (summary or {})
        parsed["format_ok"] = format_ok

        # 4. compress_succeeded: when a <compress_trigger> was injected, did
        #    the model emit action=compress with a valid <summary>? Failure =
        #    trigger ignored or summary unparseable. Only meaningful when
        #    compress_telemetry is set.
        if _compress_telemetry is not None:
            parsed["compress_succeeded"] = (
                action == "compress"
                and "summary" in (parsed.get("payload") or {})
                and "time_range" in (parsed["payload"]["summary"] or {})
            )
        else:
            parsed["compress_succeeded"] = None  # N/A this step

        # v12.6: surface the prompt actually used (build_single_step_messages
        # output, possibly extended with the recall multi-turn shape) so RL
        # loss-time reconstruction can rebuild the exact context the policy
        # conditioned on. Captured AFTER recall extension so multi-turn
        # recall samples carry the [..., assistant(tool_call), user(tool_result)]
        # tail used at the second generate.
        if locals().get("recall_messages") is not None:
            parsed["step_messages"] = recall_messages
        else:
            parsed["step_messages"] = self._last_step_messages

        return parsed
