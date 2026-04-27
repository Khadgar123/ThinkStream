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
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional

from thinkstream.data.agent_protocol import (
    AGENT_CHUNK_SEC,
    FRAMES_PER_CHUNK,
    SYSTEM_PROMPT,
    VISUAL_WINDOW_CHUNKS,
    build_user_content,
    format_memory_block,
    parse_agent_output,
)

logger = logging.getLogger(__name__)

# Token-based compression trigger (matches data construction config.py)
RECENT_THINKS_TOKEN_BUDGET = 600
COMPRESS_TRIGGER_RATIO = 0.8
COMPRESS_TOKEN_THRESHOLD = int(RECENT_THINKS_TOKEN_BUDGET * COMPRESS_TRIGGER_RATIO)  # 480
COMPRESS_RANGE_MIN = 4


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

    def count_recent_tokens(self) -> int:
        """Count total tokens in recent_thinks."""
        total = 0
        for item in self.recent_thinks:
            text = item.get("text", "")
            if self._tokenizer:
                total += len(self._tokenizer.encode(text, add_special_tokens=False))
            else:
                total += len(text) // 4
        return total

    def should_compress(self) -> bool:
        """Trigger compression when recent_thinks reach 80% of token budget.

        v9.4.2: also fire compress if individual thinks are pathologically
        long (e.g., 600+ tok each — verbose-model drift) and we have ≥2
        thinks. Without this, a model that emits one 1000-tok think then
        another would carry 2000 tok in recent_thinks while waiting for
        the COMPRESS_RANGE_MIN=4 threshold; that buffer silently inflates
        the next chunk's prompt and risks model_max_length overflow.
        """
        n_tokens = self.count_recent_tokens()
        n_thinks = len(self.recent_thinks)
        # Standard SFT-aligned trigger
        if n_tokens >= COMPRESS_TOKEN_THRESHOLD and n_thinks >= COMPRESS_RANGE_MIN:
            return True
        # v9.4.2 emergency fire: tokens far over budget, even with <4 thinks.
        # Threshold = 1.5× normal trigger; fires at len ≥ 2 to leave at
        # least one think for compress() to do anything meaningful.
        if n_tokens >= int(COMPRESS_TOKEN_THRESHOLD * 1.5) and n_thinks >= 2:
            return True
        return False

    def compress(self, summary: Dict, compressed_chunks: Optional[List[int]] = None):
        """Replace specified thinks with summary in model context.

        Raw thinks stay in _retrieval_archive for recall.

        v9.4.2: cap incoming summary text at 200 tokens to match SFT data
        construction (config.py:SUMMARY_TOKENS_MAX=180 + slack). Without
        this, a verbose model could append 400-600 tok summaries; with 5
        such segments stacked before merge fires, compressed_segments
        zone alone reaches ~3000 tok — silently overflowing model_max_length.
        Cap is matched to the SUMMARY_TOKENS_MAX constant so eval renders
        segments at the same size SFT trained on.
        """
        if compressed_chunks is not None:
            chunk_set = set(compressed_chunks)
            self.recent_thinks = [
                t for t in self.recent_thinks if t["chunk"] not in chunk_set
            ]
        else:
            self.recent_thinks = self.recent_thinks[COMPRESS_RANGE_MIN:]
        # Cap summary text BEFORE storing
        if self._tokenizer and isinstance(summary.get("text"), str):
            text = summary["text"]
            ids = self._tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > 200:
                summary = dict(summary)  # don't mutate caller's dict
                summary["text"] = self._tokenizer.decode(ids[:200])
                summary["_truncated"] = True
        self.compressed_segments.append(summary)
        # Merge oldest two if over limit (MAX_COMPRESSED_SEGMENTS=5).
        # v9.4.2: keep merged-text cap at 200 tokens (the SFT-baked value).
        # Total compressed zone = 5 segs × 200 = 1000 tok worst case, well
        # within budget after the visual fix freed ~8K tokens. Earlier
        # tightening to 150 was over-defensive: SFT data was constructed
        # with the 200-cap, so eval at <200 is fine but >200 would OOD.
        while len(self.compressed_segments) > 5:
            seg_a = self.compressed_segments.pop(0)
            seg_b = self.compressed_segments.pop(0)
            combined = f'{seg_a["text"]} {seg_b["text"]}'
            if self._tokenizer:
                ids = self._tokenizer.encode(combined, add_special_tokens=False)
                if len(ids) > 200:
                    combined = self._tokenizer.decode(ids[:200])
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


# format_memory_block, build_user_content, parse_agent_output are imported
# from thinkstream.data.agent_protocol (single source of truth).


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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Simple Retrieval (for recall action)
# ---------------------------------------------------------------------------


def bm25_retrieve(
    query: Dict,
    archive: List[Dict],
    max_results: int = 4,
) -> Dict:
    """BM25-based retrieval from archive.

    Uses rank_bm25 if available, falls back to keyword overlap.
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
        # 1. Apply chat template
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # 2. Process with vision
        inputs = processor(
            text=[text_prompt],
            return_tensors="pt",
        )

        # 3. Compute position IDs
        inputs_for_rope = dict(inputs)
        inputs_for_rope["video_chunk_size"] = 2.0  # AGENT_CHUNK_SEC
        inputs["position_ids"] = compute_position_ids(
            inputs_for_rope, processor, model_type,
        )

        inputs = inputs.to(device)

        # 4. Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=kwargs.get("temperature", 0.7),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 0.95),
        )

        # 5. Decode (only new tokens)
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
        self.tokenizer = tokenizer       # v9.4.2: needed for telemetry token counts
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
        # Try relative path under video_root, fallback to basename
        if self.video_root:
            try:
                rel = vp.relative_to(Path(self.video_root))
                stem = rel.with_suffix("")
                frame_dir = Path(self.frames_root) / stem
            except ValueError:
                frame_dir = Path(self.frames_root) / vp.stem
        else:
            frame_dir = Path(self.frames_root) / vp.stem

        if not frame_dir.exists():
            return None

        # ffmpeg frame_000001.jpg corresponds to t=0s, frame_000002.jpg to t=1s, ...
        start_frame = int(video_start) + 1
        end_frame = int(video_end) + 1
        frame_paths = []
        for i in range(start_frame, end_frame + 1):
            fp = frame_dir / f"frame_{i:06d}.jpg"
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
        # the range and inventing one. We pick the oldest COMPRESS_RANGE_MIN
        # thinks (FIFO, deterministic — same policy as pass2_rollout) and
        # encode their span. The model only learns to write the summary
        # text given a fixed range, never to choose the range itself.
        compress_trigger = ""
        # v9.4.2: telemetry for streaming eval — record state at the moment
        # compression FIRES so eval can stat: how many thinks were buffered
        # (vs the 480-tok / 4-think threshold) and which chunks got rolled
        # into the summary. Set on parsed below.
        _compress_telemetry = None
        if self.compress_mode == "system" and self.memory.should_compress():
            oldest = self.memory.recent_thinks[:COMPRESS_RANGE_MIN]
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

        # 5. Generate
        output_text = self.generate_fn(
            messages=messages,
            processor=self.processor,
            max_new_tokens=self.max_new_tokens,
            **generate_kwargs,
        )

        # 6. Parse output
        parsed = parse_agent_output(output_text)

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

                # Build recalled_frames info
                recalled_frames = None
                if returned_chunks and recall_result.get("source") == "historical_frames":
                    t_start = returned_chunks[0] * AGENT_CHUNK_SEC
                    t_end = (returned_chunks[-1] + 1) * AGENT_CHUNK_SEC
                    recalled_frames = {
                        "time_range": [int(t_start), int(t_end)],
                        "n_frames": len(returned_chunks) * FRAMES_PER_CHUNK,
                        "source": "historical_frames",
                    }

                # Build recall_response input. The pending question is
                # already expressed via the <queries> block (entry with
                # empty answers list); training data shows pending status
                # this same way and never via memory.pending_questions.
                recall_messages = build_single_step_messages(
                    snapshot,
                    chunk_idx,
                    video_path,
                    user_input="Continue following the protocol to respond.",
                    queries=self.memory.queries,
                    recalled_frames=recalled_frames,
                    recall_result=recall_result,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )

                # Second generate (allow_recall=False to prevent infinite loop)
                recall_gen_kwargs = dict(generate_kwargs)
                recall_gen_kwargs["allow_recall"] = False
                recall_output_text = self.generate_fn(
                    messages=recall_messages,
                    processor=self.processor,
                    max_new_tokens=self.max_new_tokens,
                    **recall_gen_kwargs,
                )

                recall_parsed = parse_agent_output(recall_output_text)
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

        return parsed
