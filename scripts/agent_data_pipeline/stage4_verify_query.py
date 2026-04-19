"""
Stage 4: Recall Query Verification

For each recall-positive episode:
1. Build FAISS index from the video's segment text embeddings
2. Test each query candidate against the index
3. Rerank top-20 → top-5 → check top-3 hit on gold support
4. Select the shortest query that hits
5. If all candidates fail, attempt query repair
6. Mark episodes that need teacher (397B) repair

Usage:
    python -m scripts.agent_data_pipeline.stage4_verify_query \
        [--repair_api_base http://localhost:8000/v1]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    EPISODE_DENSE_PATH,
    EPISODE_VERIFIED_PATH,
    RECALL_TOPK,
    TEACHER_QUERY_REPAIR_SYSTEM,
    ensure_dirs,
)
from .utils import (
    cosine_similarity_np,
    load_embedding,
    load_segment_archive,
    read_jsonl,
    temporal_overlap,
    write_jsonl,
)

logger = logging.getLogger(__name__)


# ===================================================================
# FAISS index management
# ===================================================================


class SegmentIndex:
    """FAISS-backed index for a single video's segment embeddings."""

    def __init__(self, segments: List[Dict]):
        self.segments = segments
        self.segment_ids = []
        self.embeddings = None
        self.index = None
        self._build()

    def _build(self):
        emb_list = []
        ids = []
        for seg in self.segments:
            emb_path = seg.get("text_emb_path", "")
            if not emb_path or not Path(emb_path).exists():
                continue
            try:
                emb = load_embedding(emb_path)
                emb_list.append(emb.flatten())
                ids.append(seg["segment_id"])
            except Exception:
                continue

        if not emb_list:
            logger.warning("No embeddings available for index")
            return

        self.segment_ids = ids
        self.embeddings = np.vstack(emb_list).astype("float32")

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = self.embeddings / norms

        try:
            import faiss

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        except ImportError:
            logger.warning("FAISS not available, using brute-force search")
            self.index = None

    def search(self, query_emb: np.ndarray, k: int = 20) -> List[Tuple[str, float]]:
        """Search for top-k most similar segments.

        Returns list of (segment_id, score) tuples.
        """
        if self.embeddings is None or len(self.segment_ids) == 0:
            return []

        query = query_emb.flatten().astype("float32")
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        if self.index is not None:
            import faiss
            scores, indices = self.index.search(query.reshape(1, -1), min(k, len(self.segment_ids)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    results.append((self.segment_ids[idx], float(score)))
            return results
        else:
            # Brute-force fallback
            scores = self.embeddings @ query
            top_indices = np.argsort(scores)[::-1][:k]
            return [(self.segment_ids[i], float(scores[i])) for i in top_indices]

    def get_segment(self, segment_id: str) -> Optional[Dict]:
        for seg in self.segments:
            if seg["segment_id"] == segment_id:
                return seg
        return None


# ===================================================================
# Query encoding
# ===================================================================


_text_encoder = None


def _get_text_encoder():
    global _text_encoder
    if _text_encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _text_encoder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct")
        except ImportError:
            try:
                from FlagEmbedding import FlagModel
                _text_encoder = FlagModel("Alibaba-NLP/gte-Qwen2-7B-instruct")
            except ImportError:
                logger.error("No text encoder available (install sentence-transformers or FlagEmbedding)")
                return None
    return _text_encoder


def encode_query(query_text: str) -> Optional[np.ndarray]:
    """Encode a query string into an embedding vector."""
    encoder = _get_text_encoder()
    if encoder is None:
        return None
    emb = encoder.encode(query_text)
    return np.array(emb).flatten()


# ===================================================================
# Coverage computation
# ===================================================================


def compute_support_coverage(
    retrieved_ids: List[str],
    gold_ids: List[str],
    segments: List[Dict],
) -> float:
    """Compute temporal overlap between retrieved and gold support segments."""
    def get_spans(seg_ids):
        spans = []
        seg_map = {s["segment_id"]: s for s in segments}
        for sid in seg_ids:
            if sid in seg_map:
                spans.append((seg_map[sid]["start_ms"], seg_map[sid]["end_ms"]))
        return spans

    retrieved_spans = get_spans(retrieved_ids)
    gold_spans = get_spans(gold_ids)

    return temporal_overlap(retrieved_spans, gold_spans)


# ===================================================================
# Query verification for a single episode
# ===================================================================


def verify_episode_queries(
    episode: Dict,
    index: SegmentIndex,
    segments: List[Dict],
) -> Dict:
    """Test all query candidates and select the best one."""
    if not episode.get("need_recall", False):
        return episode

    gold_support = set(episode.get("support_segment_ids", []))
    candidates = episode.get("query_candidates", [])

    if not candidates:
        episode["query_verification"] = "no_candidates"
        return episode

    best_query = None
    best_retrieved = None
    best_coverage = 0.0

    for candidate in candidates:
        query_text = candidate.get("query", "")
        if not query_text:
            continue

        query_emb = encode_query(query_text)
        if query_emb is None:
            continue

        # Search
        top_results = index.search(query_emb, k=20)
        top_ids = [sid for sid, _ in top_results[:RECALL_TOPK]]

        # Check hit
        retrieved_set = set(top_ids)
        hit = len(retrieved_set & gold_support) > 0

        if not hit:
            continue

        # Check coverage
        coverage = compute_support_coverage(top_ids, list(gold_support), segments)
        if coverage >= 0.5 and (best_query is None or len(query_text) < len(best_query.get("query", ""))):
            best_query = candidate
            best_retrieved = top_ids
            best_coverage = coverage

    if best_query is not None:
        episode["gold_query"] = best_query
        episode["gold_retrieved_segment_ids"] = best_retrieved
        episode["query_verification"] = "direct_hit"
        episode["query_coverage"] = best_coverage
    else:
        episode["query_verification"] = "miss"
        episode["needs_query_repair"] = True

    return episode


# ===================================================================
# Query repair (small model)
# ===================================================================


def repair_query_small_model(
    episode: Dict,
    segments: List[Dict],
    index: SegmentIndex,
) -> Dict:
    """Attempt to repair a failed query using simple heuristics."""
    gold_support = set(episode.get("support_segment_ids", []))

    # Collect anchor terms from gold support segments
    anchor_entities = set()
    anchor_actions = set()
    for seg in segments:
        if seg["segment_id"] in gold_support:
            anchor_entities.update(seg.get("entity_tags", []))
            anchor_actions.update(seg.get("action_tags", []))

    # Generate repair candidates
    repair_candidates = []
    entities = list(anchor_entities)[:3]
    actions = list(anchor_actions)[:2]

    if entities and actions:
        repair_candidates.append(" ".join(entities[:2] + actions[:1]))
        repair_candidates.append(" ".join(entities[:1] + actions[:2]))
        repair_candidates.append(" ".join(entities + actions))

    for rc_text in repair_candidates:
        rc_emb = encode_query(rc_text)
        if rc_emb is None:
            continue

        top_results = index.search(rc_emb, k=20)
        top_ids = [sid for sid, _ in top_results[:RECALL_TOPK]]
        hit = len(set(top_ids) & gold_support) > 0

        if hit:
            coverage = compute_support_coverage(top_ids, list(gold_support), segments)
            if coverage >= 0.5:
                episode["gold_query"] = {
                    "query": rc_text,
                    "time_bias": episode.get("query_candidates", [{}])[0].get("time_bias", "past_far"),
                    "target": episode.get("query_candidates", [{}])[0].get("target", "event"),
                }
                episode["gold_retrieved_segment_ids"] = top_ids
                episode["query_verification"] = "small_model_repair"
                episode["query_coverage"] = coverage
                episode.pop("needs_query_repair", None)
                return episode

    # Still failed
    episode["needs_teacher_repair"] = True
    return episode


# ===================================================================
# Query repair (teacher 397B)
# ===================================================================


def repair_query_teacher(
    episode: Dict,
    segments: List[Dict],
    index: SegmentIndex,
    api_base: str,
    model: str,
) -> Dict:
    """Use the 397B teacher to generate better query candidates."""
    from .stage2_teacher import call_teacher_api

    # Build context
    gold_support = episode.get("support_segment_ids", [])
    seg_map = {s["segment_id"]: s for s in segments}
    support_summary = "; ".join(
        seg_map[sid].get("dense_caption", "")[:40]
        for sid in gold_support if sid in seg_map
    )

    user_prompt = (
        f"question = {episode.get('question', '')}\n"
        f"visible_context = 最近24秒的视频片段\n"
        f"gold_support = {support_summary}\n"
        f"failure_reason = 原始 query candidates 均无法命中 gold support\n\n"
        "输出：\n"
        '{\n  "candidates": [\n'
        '    {"query":"...", "time_bias":"past_far", "target":"..."},\n'
        '    {"query":"...", "time_bias":"past_far", "target":"..."},\n'
        '    {"query":"...", "time_bias":"past_far", "target":"..."}\n'
        '  ]\n}'
    )

    raw = call_teacher_api(
        system_prompt=TEACHER_QUERY_REPAIR_SYSTEM,
        user_prompt=user_prompt,
        api_base=api_base,
        model=model,
        max_tokens=512,
    )

    if raw is None:
        return episode

    try:
        import re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            data = json.loads(match.group())
            new_candidates = data.get("candidates", [])
        else:
            return episode
    except (json.JSONDecodeError, ValueError):
        return episode

    # Test new candidates
    gold_set = set(gold_support)
    for cand in new_candidates:
        query_text = cand.get("query", "")
        if not query_text:
            continue
        query_emb = encode_query(query_text)
        if query_emb is None:
            continue

        top_results = index.search(query_emb, k=20)
        top_ids = [sid for sid, _ in top_results[:RECALL_TOPK]]
        hit = len(set(top_ids) & gold_set) > 0

        if hit:
            coverage = compute_support_coverage(top_ids, list(gold_set), segments)
            if coverage >= 0.5:
                episode["gold_query"] = cand
                episode["gold_retrieved_segment_ids"] = top_ids
                episode["query_verification"] = "teacher_repair"
                episode["query_coverage"] = coverage
                episode.pop("needs_query_repair", None)
                episode.pop("needs_teacher_repair", None)
                return episode

    # Final failure
    episode["query_verification"] = "failed"
    return episode


# ===================================================================
# Batch processing
# ===================================================================


def verify_all_episodes(
    repair_api_base: Optional[str] = None,
    repair_model: str = "Qwen/Qwen3.5-397B-A22B-FP8",
) -> Dict[str, int]:
    """Run query verification on all dense episodes.

    Returns stats dict.
    """
    ensure_dirs()

    if not EPISODE_DENSE_PATH.exists():
        logger.error("No dense episodes at %s", EPISODE_DENSE_PATH)
        return {}

    episodes = read_jsonl(EPISODE_DENSE_PATH)
    logger.info("Stage 4: Verifying queries for %d episodes", len(episodes))

    # Cache indexes per video
    index_cache: Dict[str, SegmentIndex] = {}
    segment_cache: Dict[str, List[Dict]] = {}

    stats = {"direct_hit": 0, "small_repair": 0, "teacher_repair": 0, "failed": 0, "skip": 0}

    for ep in episodes:
        if not ep.get("need_recall", False):
            stats["skip"] += 1
            continue

        video_id = ep["video_id"]
        if video_id not in segment_cache:
            segment_cache[video_id] = load_segment_archive(video_id)
            index_cache[video_id] = SegmentIndex(segment_cache[video_id])

        index = index_cache[video_id]
        segments = segment_cache[video_id]

        # Step 1: Direct verification
        ep = verify_episode_queries(ep, index, segments)

        if ep.get("query_verification") == "direct_hit":
            stats["direct_hit"] += 1
            continue

        # Step 2: Small model repair
        ep = repair_query_small_model(ep, segments, index)
        if ep.get("query_verification") == "small_model_repair":
            stats["small_repair"] += 1
            continue

        # Step 3: Teacher repair (if API available)
        if repair_api_base and ep.get("needs_teacher_repair"):
            ep = repair_query_teacher(ep, segments, index, repair_api_base, repair_model)
            if ep.get("query_verification") == "teacher_repair":
                stats["teacher_repair"] += 1
                continue

        stats["failed"] += 1

    write_jsonl(episodes, EPISODE_VERIFIED_PATH)

    logger.info("Stage 4 results: %s", stats)
    return stats


# ===================================================================
# CLI
# ===================================================================


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Query verification")
    parser.add_argument("--repair_api_base", default=None)
    parser.add_argument("--repair_model", default="Qwen/Qwen3.5-397B-A22B-FP8")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = verify_all_episodes(args.repair_api_base, args.repair_model)

    print(f"\nStage 4 Summary:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
