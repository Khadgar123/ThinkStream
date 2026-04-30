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
from pathlib import Path
from typing import Callable, Dict, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    SYSTEM_PROMPT_V12,
    VISUAL_WINDOW_CHUNKS,
    build_user_content,
    format_memory_block,
    parse_agent_output_v12,
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


# ---------------------------------------------------------------------------
# Memory State (mirrors pass2_rollout.py:MemoryState)
# ---------------------------------------------------------------------------


class MemoryState:
    """Tracks the agent's text memory at each timestep.

    Mirrors scripts/agent_data_v5/pass2_rollout.py:MemoryState exactly.
    Text memory covers LONGER time than visual window.
    """

    def __init__(self, tokenizer=None):
        self.compressed_segments: List[Dict] = []
        self.recent_thinks: List[Dict] = []
        self._retrieval_archive: List[Dict] = []
        self._tokenizer = tokenizer

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

    def add_think(self, chunk_idx: int, think_text: str):
        """Add think to memory immediately."""
        time_start = chunk_idx * AGENT_CHUNK_SEC
        time_end = time_start + AGENT_CHUNK_SEC
        item = {
            "chunk": chunk_idx,
            "time": f"{int(time_start)}-{int(time_end)}",
            "text": think_text,
        }
        self.recent_thinks.append(item)
        self._retrieval_archive.append(item)

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
        if compressed_chunks is not None:
            chunk_set = set(compressed_chunks)
            self.recent_thinks = [
                t for t in self.recent_thinks if t["chunk"] not in chunk_set
            ]
        else:
            n = select_compress_range_by_tokens(
                self.recent_thinks,
                token_count_fn=self._token_count,
            )
            self.recent_thinks = self.recent_thinks[n:] if n > 0 else self.recent_thinks
        if self._tokenizer and isinstance(summary.get("text"), str):
            ids = self._tokenizer.encode(summary["text"], add_special_tokens=False)
            if len(ids) > SUMMARY_TOKENS_MAX:
                summary = dict(summary)
                summary["text"] = self._tokenizer.decode(ids[:SUMMARY_TOKENS_MAX])
                summary["_truncated"] = True
        self.compressed_segments.append(summary)
        # Merge oldest two if over MAX_COMPRESSED_SEGMENTS=5.
        while len(self.compressed_segments) > 5:
            seg_a = self.compressed_segments.pop(0)
            seg_b = self.compressed_segments.pop(0)
            combined = f'{seg_a["text"]} {seg_b["text"]}'
            if self._tokenizer:
                ids = self._tokenizer.encode(combined, add_special_tokens=False)
                if len(ids) > SUMMARY_TOKENS_MAX:
                    combined = self._tokenizer.decode(ids[:SUMMARY_TOKENS_MAX])
            merged = {
                "time_range": [seg_a["time_range"][0], seg_b["time_range"][1]],
                "text": combined,
                "merged": True,
                "merge_level": max(
                    seg_a.get("merge_level", 1), seg_b.get("merge_level", 1)
                ) + 1,
            }
            self.compressed_segments.insert(0, merged)

    # --- Queries tracking (matches SFT <queries> zone) ---
    # The legacy add_pending / resolve_pending pair was removed in v11.1
    # — training data had pending_questions empty across all 12,405
    # samples, so the field was reverse-OOD at inference. "Pending" is
    # now expressed by add_query() leaving `answers=[]`; once
    # answer_query() runs, the entry becomes answered. format_memory_block
    # still tolerates the legacy field via .get() for back-compat with
    # any external snapshot dumps.

    def add_query(self, question: str, ask_time: float):
        """Register a question (pending until answered)."""
        if not hasattr(self, "_queries"):
            self._queries = []
        self._queries.append({
            "question": question,
            "ask_time": ask_time,
            "answers": [],
        })

    def answer_query(self, question: str, answer: str, response_time: float):
        """Record an answer for a pending query."""
        if not hasattr(self, "_queries"):
            return
        for q in reversed(self._queries):
            if q["question"] == question:
                q["answers"].append({"text": answer, "time": response_time})
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
    min_pixels: int = 100352,
    max_pixels: int = 150528,
    frame_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """Build single-step chat messages matching training format.

    Delegates text formatting to shared agent_protocol.build_user_content.
    Uses ``SYSTEM_PROMPT_V12`` (the only protocol). The ``<tools>`` block is
    rendered by the chat_template — callers must pass ``tools=TOOLS_SCHEMA``
    when invoking ``processor.apply_chat_template``.
    """
    memory_text = format_memory_block(snapshot)
    user_content = build_user_content(
        memory_text,
        chunk_idx,
        video_path,
        user_input=user_input,
        queries=queries,
        recalled_frames=recalled_frames,
        recall_result=recall_result,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        frame_paths=frame_paths,
    )

    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_V12}]},
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


def filter_archive_by_time_range(
    archive: List[Dict], time_range, chunk_sec: float = AGENT_CHUNK_SEC,
) -> List[Dict]:
    """Restrict archive to items whose chunk overlaps [t_start, t_end].

    "with_time_range" mode = the model emits a time_range and the retriever
    pre-filters to that window before scoring. Falls back to the full
    archive when the range is missing/malformed (matches the SFT
    distribution where ~30% of queries are keyword-only by design).
    """
    tr = parse_time_range(time_range)
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
    max_results: int = 4,
) -> Dict:
    """BM25-based retrieval from archive.

    Honours `query["time_range"]` when present (filters archive to chunks
    overlapping that window); falls back to full archive on missing /
    malformed range. Uses rank_bm25 if available, else keyword overlap.
    Returns recall_result dict with text_content and returned_chunks.
    """
    query_text = query.get("query", "")
    if not query_text.strip() or not archive:
        return {
            "source": "failure",
            "time": "",
            "text_content": "No matching results found.",
            "returned_chunks": [],
        }

    archive = filter_archive_by_time_range(archive, query.get("time_range"))
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
        top_indices = [i for i in top_indices if scores[i] > 0]
    except ImportError:
        # Fallback: keyword overlap scoring
        query_words = set(query_text.lower().split())
        scored = []
        for i, text in enumerate(texts):
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
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
    returned_chunks = [item["chunk"] for item in top_items]
    text_parts = [f'[{item["time"]}] {item["text"]}' for item in top_items]

    t_start = returned_chunks[0] * AGENT_CHUNK_SEC
    t_end = (returned_chunks[-1] + 1) * AGENT_CHUNK_SEC

    return {
        "source": "historical_frames",
        "time": f"{int(t_start)}-{int(t_end)}",
        "text_content": "\n".join(text_parts),
        "returned_chunks": sorted(returned_chunks),
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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        # 2. Move to device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 3. Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
        )

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
        min_pixels: int = 100352,
        max_pixels: int = 150528,
        max_new_tokens: int = 256,
        retrieve_fn: Optional[Callable] = None,
        retriever=None,
        compress_mode: str = "system",
        frames_root: Optional[str] = None,
        video_root: Optional[str] = None,
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
                memory.should_compress() fires, system inserts a
                <compress_trigger range="t_start-t_end"/> with a fixed
                FIFO range; the model only writes the summary text.
                "self" (used by RL eval after GDPO) — system never
                inserts a trigger; the model decides autonomously when
                to emit <action>compress</action> and which range to
                summarize. Only enable "self" with an RL-tuned ckpt:
                v11 SFT samples were all C1 (system-triggered fixed
                range), so a pure-SFT model under "self" mode is OOD.
        """
        if compress_mode not in ("system", "self"):
            raise ValueError(
                f"compress_mode must be 'system' or 'self', got {compress_mode!r}"
            )
        self.generate_fn = generate_fn
        self.tokenizer = tokenizer       # needed for telemetry token counts
        self.processor = processor
        self.model_type = model_type
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
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
        self.frames_root = frames_root
        self.video_root = video_root
        self.memory = MemoryState(tokenizer=tokenizer)

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

        if not frame_dir.exists():
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
            if not q.get("answers"):
                self.memory.answer_query(q["question"], answer_text, response_time)
                return

    def step(
        self,
        chunk_idx: int,
        video_path: str,
        user_question: Optional[str] = None,
        **generate_kwargs,
    ) -> Dict:
        """Execute one agent step (one chunk).

        Returns parsed output dict with keys: think, action, payload.
        Handles compression trigger and recall orchestration internally.
        """
        # 1. Snapshot BEFORE this step
        snapshot = self.memory.snapshot(chunk_idx)

        # 1b. Register the new question (if any) into the queries log so
        # it appears in the <queries> block this step. Training data has
        # the question present in <queries> at the chunk it arrives —
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
                self.memory.add_query(user_question, ask_time)

        # 2. Check compression trigger (system-triggered, not model-triggered).
        #
        # The trigger MUST embed a `range="t_start-t_end"` attribute that
        # mirrors render_samples.py:174-180 exactly — every SFT compress
        # sample saw a trigger with this attribute, so a no-attribute
        # variant is out-of-distribution and risks (a) format drift in
        # the summary's time_range field, (b) the model failing to copy
        # the range and inventing one.
        #
        # v11.3: range size is token-driven via select_compress_range_by_tokens
        # (was hardcoded to COMPRESS_RANGE_MIN=4). Pass2 already enumerated
        # variable ranges in [4, 8] via score_range_for_compression; agent_loop
        # now matches that variability so inference and training agree on
        # the policy. The model still doesn't choose the range — it only
        # writes the summary text given a system-supplied range.
        compress_trigger = ""
        # v9.4.2: telemetry for streaming eval — record state at the moment
        # compression FIRES so eval can stat: how many thinks were buffered
        # (vs the 480-tok / 4-think threshold) and which chunks got rolled
        # into the summary. Set on parsed below.
        _compress_telemetry = None
        if self.compress_mode == "system" and self.memory.should_compress():
            n_to_compress = select_compress_range_by_tokens(
                self.memory.recent_thinks,
                token_count_fn=self.memory._token_count,
            )
            oldest = self.memory.recent_thinks[:n_to_compress] if n_to_compress > 0 else []
            if oldest:
                chunks = [t["chunk"] for t in oldest]
                t_start = min(chunks) * AGENT_CHUNK_SEC
                t_end = (max(chunks) + 1) * AGENT_CHUNK_SEC
                compress_trigger = (
                    f'<compress_trigger range="{t_start}-{t_end}"/>'
                )
                _compress_telemetry = {
                    "thinks_count_at_trigger": len(self.memory.recent_thinks),
                    "thinks_token_count": self.memory.count_recent_tokens(),
                    "compressed_chunks": chunks,
                    "trigger_chunk": chunk_idx,
                }
        # compress_mode == "self": no trigger inserted. The model is
        # expected to autonomously emit <action>compress</action> when
        # it judges memory pressure, with its own time_range in the
        # <summary>. Only used after GDPO has trained the policy to
        # pick ranges; pure-SFT ckpts will likely never compress in
        # this mode and overflow.

        # 3. Determine user_input
        user_input = ""
        if compress_trigger and not user_question:
            # Compression takes priority when no user question
            user_input = compress_trigger
        elif user_question:
            user_input = user_question

        # 4. Build single-step messages (matching training format).
        # Pass `queries=self.memory.queries` so the <queries> block is
        # populated identically to the SFT input format — both for the
        # current pending question and for any past Q/A pairs that the
        # model has already produced answers for in this video.
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
        )
        # v12.6: stash the EXACT messages used for generation so RL loss-time
        # reconstruction can replay the same prompt — see
        # thinkstream/trainer/grpo.py:_build_rollout_messages. Without this
        # the loss path conditions logprobs on a stripped-down context (no
        # <memory>, <visual_window>, <queries>) and gradient direction drifts.
        self._last_step_messages = messages

        # 5. Generate
        output_text = self.generate_fn(
            messages=messages,
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            **generate_kwargs,
        )

        # 6. Parse output
        parsed = _parse_agent_output(output_text)

        if os.environ.get("AGENT_DEBUG"):
            print(f"[AGENT_DEBUG] chunk={chunk_idx} user_input={user_input!r}")
            print(f"[AGENT_DEBUG] raw_output={output_text!r}")
            print(f"[AGENT_DEBUG] parsed action={parsed['action']!r} think_len={len(parsed['think'])}")

        # 7. Update memory state based on action
        if parsed["think"]:
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
                tr = summary["time_range"]
                compressed_chunks = []
                for t in self.memory.recent_thinks:
                    chunk_start = t["chunk"] * AGENT_CHUNK_SEC
                    chunk_end = chunk_start + AGENT_CHUNK_SEC
                    if chunk_start >= tr[0] and chunk_end <= tr[1]:
                        compressed_chunks.append(t["chunk"])
                self.memory.compress(summary, compressed_chunks=compressed_chunks)

        elif parsed["action"] == "recall":
            # Orchestrate recall: retrieve → build recall_response input → second generate
            query = parsed["payload"].get("query", {})
            if query:
                recall_result = self.retriever(
                    query, self.memory.retrieval_archive
                )
                returned_chunks = recall_result.get("returned_chunks", [])

                # Build recalled_frames info (including frame_paths so we
                # don't fallback to full-video decoding in recall_response).
                recalled_frames = None
                if returned_chunks and recall_result.get("source") == "historical_frames":
                    t_start = returned_chunks[0] * AGENT_CHUNK_SEC
                    t_end = (returned_chunks[-1] + 1) * AGENT_CHUNK_SEC
                    recalled_frames = {
                        "time_range": [int(t_start), int(t_end)],
                        "n_frames": len(returned_chunks) * FRAMES_PER_CHUNK,
                        "source": "historical_frames",
                    }
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
                        rf_paths = []
                        if frame_dir.exists():
                            # v12.6 fix: same chunk×FRAMES_PER_CHUNK convention
                            # used everywhere else (pass1a, _get_frame_paths,
                            # streaming_vllm). Old code used seconds-based
                            # offsets which were off-by-half under FPS=2.
                            for rc in returned_chunks:
                                for fi in range(FRAMES_PER_CHUNK):
                                    fnum = rc * FRAMES_PER_CHUNK + fi + 1
                                    fp = frame_dir / f"frame_{fnum:06d}.jpg"
                                    if fp.exists():
                                        rf_paths.append(str(fp))
                        if rf_paths:
                            recalled_frames["frame_paths"] = rf_paths

                # v12.6 fix: build true multi-turn recall prompt matching
                # SFT shape B (pass5_messages.py:212-260).
                # Old behavior REBUILT a fresh single-turn prompt with
                # recall_result inlined into the same user content, dropping
                # the model's own recall tool_call from context — train/infer
                # divergence. New behavior:
                #   [system, user(chunk N), assistant(recall tool_call),
                #    user(recall_result + recalled_frames)] → generate answer
                # This is byte-identical to the SFT trajectory the model saw.
                import json as _json
                recall_messages = list(messages)             # [system, user(chunk N)]
                recall_messages.append({                      # model's own recall turn
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                })
                tool_user_content = []
                if recalled_frames:
                    rf_header = _json.dumps({
                        "time_range": recalled_frames["time_range"],
                        "source": recalled_frames.get("source", "historical_frames"),
                        "n_frames": recalled_frames["n_frames"],
                    })
                    tool_user_content.append({
                        "type": "text",
                        "text": f"<recalled_frames>{rf_header}</recalled_frames>",
                    })
                    if "frame_paths" in recalled_frames:
                        tool_user_content.append({
                            "type": "video",
                            "video": recalled_frames["frame_paths"],
                        })
                rr_json = _json.dumps({
                    "source": recall_result.get("source", ""),
                    "time": recall_result.get("time", ""),
                    "text": recall_result.get(
                        "text_content", recall_result.get("text", "")
                    ),
                }, ensure_ascii=False)
                tool_user_content.append({
                    "type": "text",
                    "text": f"<recall_result>{rr_json}</recall_result>",
                })
                recall_messages.append({
                    "role": "user", "content": tool_user_content,
                })

                # Second generate (allow_recall=False to prevent infinite loop)
                recall_gen_kwargs = dict(generate_kwargs)
                recall_gen_kwargs["allow_recall"] = False
                recall_output_text = self.generate_fn(
                    messages=recall_messages,
                    processor=self.processor,
                    max_new_tokens=self.max_new_tokens,
                    **recall_gen_kwargs,
                )

                recall_parsed = _parse_agent_output(recall_output_text)
                # recall_response has NO think (observation was already
                # emitted in sample1 for this same chunk_idx).

                # Merge recall results into parsed output
                parsed["recall_step2"] = recall_parsed
                parsed["recall_result"] = recall_result
                # Override action to the final action (response or silent)
                if recall_parsed["action"] in ("response", "silent"):
                    parsed["final_action"] = recall_parsed["action"]
                    parsed["final_payload"] = recall_parsed["payload"]
                # If the recall second pass emitted a response, log the
                # answer against the most recent unanswered query so the
                # next chunk's <queries> block carries it forward.
                if recall_parsed["action"] == "response":
                    answer_text = recall_parsed["payload"].get("response", "")
                    self._record_answer(answer_text, chunk_idx)

        elif parsed["action"] == "response":
            # Log the answer in the queries log so it shows up in the
            # next chunk's <queries> block. We attribute it to the most
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
        # 1. prompt_text_token_count: text-only zones (system + memory + queries
        #    + recall + user_input). Visual frames excluded — they're a fixed
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
        format_ok = bool(parsed.get("think")) and action in VALID_ACTIONS
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
