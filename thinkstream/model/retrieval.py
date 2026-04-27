"""Retrieval backends for StreamingAgentLoop's recall step.

Two implementations, both matching the same callable signature
expected by agent_loop's `retrieve_fn` slot:

  retriever(query: Dict, archive: List[Dict]) -> Dict
      Same signature as bm25_retrieve in agent_loop.py — returns a
      recall_result dict with text_content / returned_chunks / time.

  retriever.index_chunk(chunk_idx, video_path, think_text) -> None
      Optional indexing hook called by agent_loop after each chunk's
      think is added to memory. BM25Retriever no-ops here (text
      archive lives in MemoryState already). HybridRetriever stores
      a per-chunk visual embedding for dense visual scoring.

Why hybrid:
  Pure BM25 fails when the model's <think> uses different vocab from
  the question (e.g., think writes "person in dark jacket" but query
  asks "guy wearing coat"). Visual similarity bypasses this lexical
  gap. Combining them gives more robust top-K — typical 2-5pp gain on
  Backward-Tracing tasks where retrieval matters.

Cost:
  Hybrid loads SigLIP (~600MB), adds ~5ms/chunk indexing and
  ~10ms/retrieval. Negligible vs the ~1s/chunk Qwen3-VL forward,
  but only enable if you actually want dense scoring (the BM25
  baseline is already in agent_loop and works).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

AGENT_CHUNK_SEC = 2.0
FRAMES_PER_CHUNK = 2


# ─── Protocol ────────────────────────────────────────────────────────────────


class Retriever(Protocol):
    def index_chunk(self, chunk_idx: int, video_path: str, think_text: str) -> None: ...
    def __call__(self, query: Dict, archive: List[Dict]) -> Dict: ...


def _empty_recall() -> Dict:
    return {
        "source": "failure",
        "time": "",
        "text_content": "No matching results found.",
        "returned_chunks": [],
    }


# ─── BM25 baseline (matches existing v11 behavior) ───────────────────────────


class BM25Retriever:
    """Stateless BM25 retriever — forwards to agent_loop.bm25_retrieve."""

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    def index_chunk(self, chunk_idx, video_path, think_text):
        pass  # think text is already in MemoryState._retrieval_archive

    def __call__(self, query, archive):
        from thinkstream.model.agent_loop import bm25_retrieve
        return bm25_retrieve(query, archive, max_results=self.max_results)


# ─── Hybrid: BM25 + dense visual ─────────────────────────────────────────────


class HybridRetriever:
    """BM25 (text) + dense visual similarity (frames) hybrid.

    Stores a per-chunk L2-normalized visual embedding. At retrieval
    time:
        score = alpha * minmax(BM25) + (1 - alpha) * (cos(q, v) + 1)/2
    Both terms are mapped to [0, 1] so alpha is a real interpolation
    weight (alpha=1 collapses to pure BM25, alpha=0 to pure visual).

    The encode_image_fn / encode_text_fn callables abstract over the
    embedding model (default factory: SigLIP). Pass your own pair to
    swap in CLIP, EVA-CLIP, InternVideo, etc.
    """

    def __init__(
        self,
        encode_image_fn: Callable,
        encode_text_fn: Callable,
        *,
        alpha: float = 0.5,
        max_results: int = 4,
        device: str = "cuda",
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.encode_image = encode_image_fn
        self.encode_text = encode_text_fn
        self.alpha = alpha
        self.max_results = max_results
        self.device = device
        self.chunk_embeddings: Dict[int, torch.Tensor] = {}

    def _extract_frames(self, video_path: str, chunk_idx: int):
        """Pull FRAMES_PER_CHUNK frames from chunk's [t0, t1] range."""
        try:
            from decord import VideoReader, cpu
        except ImportError:
            logger.warning("decord not installed — hybrid retriever can't index visual")
            return None
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = float(vr.get_avg_fps())
            t0 = chunk_idx * AGENT_CHUNK_SEC
            t1 = t0 + AGENT_CHUNK_SEC
            f0 = int(t0 * fps)
            f1 = min(int(t1 * fps), len(vr) - 1)
            if f0 >= len(vr):
                return None
            if f1 <= f0:
                f1 = f0 + 1
            indices = np.linspace(f0, f1, FRAMES_PER_CHUNK).astype(int).tolist()
            return list(vr.get_batch(indices).asnumpy())
        except Exception as e:
            logger.warning("Frame extract failed (chunk=%d): %s", chunk_idx, e)
            return None

    def index_chunk(self, chunk_idx, video_path, think_text):
        if chunk_idx in self.chunk_embeddings:
            return
        frames = self._extract_frames(video_path, chunk_idx)
        if frames is None:
            return
        with torch.no_grad():
            emb = self.encode_image(frames)             # (n, D)
            emb = emb.mean(dim=0)
            emb = emb / (emb.norm() + 1e-8)
        self.chunk_embeddings[chunk_idx] = emb.detach().cpu()

    def __call__(self, query, archive):
        query_text = query.get("query", "")
        if not query_text.strip() or not archive:
            return _empty_recall()

        # 1. BM25 (text)
        try:
            from rank_bm25 import BM25Okapi
            corpus = [t.get("text", "").lower().split() for t in archive]
            bm25 = BM25Okapi(corpus)
            bm25_scores = bm25.get_scores(query_text.lower().split())
        except ImportError:
            qw = set(query_text.lower().split())
            bm25_scores = np.array(
                [len(qw & set(t.get("text", "").lower().split())) for t in archive],
                dtype=float,
            )

        bm25_norm = self._minmax(bm25_scores)

        # 2. Visual similarity
        vis_norm = np.zeros(len(archive))
        if self.chunk_embeddings:
            with torch.no_grad():
                qemb = self.encode_text(query_text).squeeze()
                qemb = qemb / (qemb.norm() + 1e-8)
                qemb_cpu = qemb.detach().cpu()
            for i, item in enumerate(archive):
                cemb = self.chunk_embeddings.get(item["chunk"])
                if cemb is not None:
                    vis_norm[i] = (float(qemb_cpu @ cemb) + 1.0) / 2.0

        # 3. Combine
        scores = self.alpha * bm25_norm + (1.0 - self.alpha) * vis_norm
        order = sorted(
            (i for i in range(len(scores)) if scores[i] > 0),
            key=lambda i: -scores[i],
        )[: self.max_results]

        if not order:
            return _empty_recall()

        top = [archive[i] for i in order]
        chunks = sorted(item["chunk"] for item in top)
        text_parts = [f'[{item["time"]}] {item["text"]}' for item in top]
        return {
            "source": "historical_frames",
            "time": f"{int(chunks[0] * AGENT_CHUNK_SEC)}-{int((chunks[-1] + 1) * AGENT_CHUNK_SEC)}",
            "text_content": "\n".join(text_parts),
            "returned_chunks": chunks,
            "_score_breakdown": {
                "alpha": self.alpha,
                "bm25_top": float(bm25_norm[order[0]]),
                "vis_top": float(vis_norm[order[0]]),
            },
        }

    @staticmethod
    def _minmax(s: np.ndarray) -> np.ndarray:
        if s.size == 0:
            return s
        lo, hi = float(s.min()), float(s.max())
        if hi - lo < 1e-8:
            return np.zeros_like(s)
        return (s - lo) / (hi - lo)


# ─── Factory ─────────────────────────────────────────────────────────────────


def _make_siglip_encoders(model_path: str, device: str):
    """Load SigLIP image+text encoders, return (encode_image, encode_text)."""
    from transformers import AutoModel, AutoProcessor

    logger.info("Loading SigLIP %s for hybrid retrieval", model_path)
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32).to(device)
    model.eval()

    @torch.no_grad()
    def encode_image(frames):
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        return model.get_image_features(**inputs)

    @torch.no_grad()
    def encode_text(text):
        inputs = proc(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(device)
        return model.get_text_features(**inputs)

    return encode_image, encode_text


def make_retriever(
    kind: str = "bm25",
    *,
    siglip_path: str = "google/siglip-base-patch16-224",
    alpha: float = 0.5,
    max_results: int = 4,
    device: str = "cuda",
) -> Retriever:
    """Build a Retriever instance.

    kind="bm25"   stateless, no extra model. Existing v11 baseline.
    kind="hybrid" loads SigLIP, runs BM25 + dense visual.
    """
    if kind == "bm25":
        return BM25Retriever(max_results=max_results)
    if kind == "hybrid":
        ei, et = _make_siglip_encoders(siglip_path, device)
        return HybridRetriever(
            encode_image_fn=ei,
            encode_text_fn=et,
            alpha=alpha,
            max_results=max_results,
            device=device,
        )
    raise ValueError(f"Unknown retriever kind: {kind}")


# ─── Backward-compat shim: wrap a callable as a Retriever ────────────────────


class _CallableRetriever:
    """Wrap a plain (query, archive) -> dict callable as a Retriever."""

    def __init__(self, fn: Callable, max_results: int = 4):
        self.fn = fn
        self.max_results = max_results

    def index_chunk(self, chunk_idx, video_path, think_text):
        pass

    def __call__(self, query, archive):
        return self.fn(query, archive)


def coerce_retriever(arg) -> Retriever:
    """Accept either a Retriever, a callable, or None. Return a Retriever."""
    if arg is None:
        return BM25Retriever()
    if hasattr(arg, "index_chunk") and callable(arg):
        return arg
    if callable(arg):
        return _CallableRetriever(arg)
    raise TypeError(f"Cannot coerce {type(arg).__name__} into a Retriever")
