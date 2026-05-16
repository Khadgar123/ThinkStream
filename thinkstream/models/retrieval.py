"""Time-range recall backend for StreamingAgentLoop.

Recall no longer performs lexical or dense retrieval. The model emits only a
closed historical ``start_time`` / ``end_time`` interval, and the tool returns
up to ``RECALL_RETURN_CHUNKS`` chunks uniformly sampled from that interval.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Protocol

from thinkstream.data.agent_protocol import (
    RECALL_RETURN_CHUNKS,
    recall_time_string_for_chunks,
    select_recall_chunks_uniform,
)


class Retriever(Protocol):
    def index_chunk(self, chunk_idx: int, video_path: str, think_text: str) -> None: ...
    def __call__(self, query: Dict, archive: List[Dict]) -> Dict: ...


class TimeRangeRetriever:
    """Stateless recall retriever over explicit time ranges."""

    def __init__(self, max_results: int = RECALL_RETURN_CHUNKS):
        self.max_results = int(max_results)

    def clone_empty(self) -> "TimeRangeRetriever":
        return TimeRangeRetriever(max_results=self.max_results)

    def index_chunk(self, chunk_idx, video_path, think_text):
        pass

    def __call__(self, query, archive):
        from thinkstream.models.agent_loop import time_range_retrieve

        return time_range_retrieve(
            query or {},
            archive or [],
            max_results=self.max_results,
        )


def make_retriever(
    kind: str = "time_range",
    *,
    max_results: int = RECALL_RETURN_CHUNKS,
    **_: object,
) -> Retriever:
    """Build a retriever.

    Only time-range recall is supported. Legacy ranked retrieval modes were removed
    because recall tool calls no longer contain a free-text query.
    """
    mode = str(kind or "time_range").strip().lower().replace("-", "_")
    if mode in {"time_range", "timerange", "uniform", "default"}:
        return TimeRangeRetriever(max_results=max_results)
    raise ValueError(
        f"Unknown retriever kind {kind!r}; only 'time_range' is supported"
    )


class _CallableRetriever:
    """Wrap a plain ``(query, archive) -> dict`` callable as a Retriever."""

    def __init__(self, fn: Callable, max_results: int = RECALL_RETURN_CHUNKS):
        self.fn = fn
        self.max_results = int(max_results)

    def index_chunk(self, chunk_idx, video_path, think_text):
        pass

    def __call__(self, query, archive):
        out = self.fn(query or {}, archive or [])
        if isinstance(out, dict):
            chunks = select_recall_chunks_uniform(
                out.get("returned_chunks") or [],
                max_chunks=self.max_results,
            )
            out["returned_chunks"] = chunks
            if chunks and not out.get("time"):
                out["time"] = recall_time_string_for_chunks(chunks)
        return out


def coerce_retriever(arg) -> Retriever:
    """Accept a Retriever, a legacy callable, or None."""
    if arg is None:
        return TimeRangeRetriever()
    if hasattr(arg, "index_chunk") and callable(arg):
        return arg
    if callable(arg):
        return _CallableRetriever(arg)
    raise TypeError(f"Cannot coerce {type(arg).__name__} into a Retriever")
