"""Multi-scheme recall retrieval audit.

Given a set of recall samples (pass3c output or model-generation output),
evaluate every scheme's ability to retrieve the gold support chunks from
the agent's archive when fed the recall <query>{...}</query> JSON.

Schemes:
  - bm25_keyword:  query.text only, no time filter
  - bm25_time:     query.text + time_range pre-filter (existing v11 default)
  - dense_text:    sentence-transformer cosine similarity over think texts
  - hybrid:        BM25 + visual cosine (SigLIP), Reciprocal Rank Fusion

Inputs (one of):
  - --samples <pass3c.jsonl>: gold queries from training data — measures
                              the retriever's CEILING (perfect query case).
  - --predictions <gen.jsonl>: model-generated queries — measures the
                              full pipeline (model query → retriever → gold).

Caching:
  Text embeddings keyed by `(video_id, chunk_idx, sha1(text)[:8])` so a
  cache stays valid across edits as long as the underlying think text
  doesn't drift. Visual embeddings keyed by `(video_id, chunk_idx)`
  (frames are deterministic at AGENT_CHUNK_SEC granularity).

Output:
  - <out_dir>/recall_audit_summary.json  aggregate Hit@k + MRR per scheme
  - <out_dir>/recall_audit_per_sample.jsonl  per-sample retrievals
  - Optional: --wandb_project pushes a wandb table

Usage:
  python scripts/audit/recall_retrieval_audit.py \
      --samples /path/to/pass3c_samples.jsonl \
      --frames_root /path/to/frames \
      --cache_dir /path/to/audit_cache \
      --out_dir /path/to/audit_out \
      --schemes bm25_keyword,bm25_time,dense_text,hybrid \
      --top_k 1,3,5
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make `from thinkstream...` work when invoked as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger("recall_audit")

AGENT_CHUNK_SEC = 2.0


# ---------------------------------------------------------------------------
# Sample loading + query extraction
# ---------------------------------------------------------------------------

_QUERY_RE = re.compile(r"<query>\s*(\{.*?\})\s*</query>", re.DOTALL)


def _extract_query_json(text: str) -> Optional[Dict]:
    """Pull `<query>{...}</query>` JSON out of an assistant output string."""
    if not text:
        return None
    m = _QUERY_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _load_samples(path: str, mode: str) -> List[Dict]:
    """Load JSONL samples and normalize to a uniform schema.

    mode == "samples": pass3c output. Each line has output with a gold
        <query>...</query>. archive built from sample.input.memory.recent_thinks.
    mode == "predictions": eval-time generation output. Expects per-sample
        dict with a `generated_text` field; falls back to "output" if absent.
    """
    out = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            stype = row.get("sample_type") or row.get("action")
            if stype not in ("recall_query", "recall"):
                continue

            if mode == "predictions":
                output_text = row.get("generated_text") or row.get("output", "")
            else:
                output_text = row.get("output", "")
            query = _extract_query_json(output_text)
            if query is None:
                continue

            inp = row.get("input", {}) or {}
            mem = inp.get("memory") or {}
            recent_thinks = mem.get("recent_thinks") or []
            archive = []
            for t in recent_thinks:
                if not isinstance(t, dict):
                    continue
                archive.append({
                    "chunk": int(t.get("chunk", -1)),
                    "time":  t.get("time", ""),
                    "text":  t.get("text", ""),
                })
            archive = [a for a in archive if a["chunk"] >= 0 and a["text"]]
            if not archive:
                continue

            # Gold = support_chunks if carried on the sample, else fall back
            # to recall_result.returned_chunks (oracle case).
            gold = row.get("support_chunks")
            if gold is None:
                rr = inp.get("recall_result") or {}
                gold = rr.get("returned_chunks") or []
            gold = sorted(int(c) for c in gold if isinstance(c, int) or
                          (isinstance(c, str) and c.isdigit()))
            if not gold:
                continue

            out.append({
                "sample_id": row.get("sample_id") or f"line_{line_no}",
                "video_id":  row.get("video_id") or "",
                "video_path": row.get("video_path") or "",
                "chunk_idx": row.get("chunk_idx", 0),
                "archive":   archive,
                "gold":      gold,
                "query":     query,
            })
    return out


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def _text_cache_key(video_id: str, chunk: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{video_id}__c{chunk}__{h}.npy"


def _visual_cache_key(video_id: str, chunk: int) -> str:
    return f"{video_id}__c{chunk}__visual.npy"


class EmbeddingCache:
    def __init__(self, cache_dir: Optional[str], encode_fn, batch_size: int = 64):
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.encode = encode_fn
        self.batch_size = batch_size
        self._mem: Dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Optional[Path]:
        return self.dir / key if self.dir is not None else None

    def get_many(self, key_text_pairs: List[Tuple[str, str]]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        miss: List[Tuple[str, str]] = []
        for key, text in key_text_pairs:
            if key in self._mem:
                out[key] = self._mem[key]
                self.hits += 1
                continue
            p = self._path(key)
            if p is not None and p.exists():
                vec = np.load(p)
                self._mem[key] = vec
                out[key] = vec
                self.hits += 1
            else:
                miss.append((key, text))
                self.misses += 1
        if miss:
            for i in range(0, len(miss), self.batch_size):
                batch = miss[i:i + self.batch_size]
                texts = [t for _, t in batch]
                vecs = self.encode(texts)  # (B, D) np.ndarray
                for (key, _), vec in zip(batch, vecs):
                    self._mem[key] = vec
                    out[key] = vec
                    p = self._path(key)
                    if p is not None:
                        np.save(p, vec)
        return out


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def _make_text_encoder(model_name: str, device: str):
    """Sentence-transformer style text encoder. Returns encode(texts)->ndarray."""
    from transformers import AutoModel, AutoTokenizer
    import torch

    logger.info(f"Loading text encoder {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    @np.errstate(invalid="ignore")
    def encode(texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inp = tok(texts, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            out = model(**inp).last_hidden_state  # (B, L, D)
            mask = inp["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        return pooled.detach().cpu().numpy().astype(np.float32)

    return encode


def _make_visual_encoder(siglip_path: str, device: str, frames_root: Optional[str]):
    """SigLIP visual encoder over pre-extracted JPEGs from frames_root.

    Returns encode_chunk(video_id, chunk) -> ndarray | None.
    Reads frame_NNNNNN.jpg files matching the chunk's time window.
    """
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import torch

    logger.info(f"Loading SigLIP {siglip_path}")
    proc = AutoProcessor.from_pretrained(siglip_path)
    model = AutoModel.from_pretrained(siglip_path).to(device)
    model.eval()

    def encode_chunk(video_id: str, chunk_idx: int) -> Optional[np.ndarray]:
        if not frames_root:
            return None
        frame_dir = Path(frames_root) / video_id
        if not frame_dir.exists():
            return None
        t0 = chunk_idx * AGENT_CHUNK_SEC
        t1 = t0 + AGENT_CHUNK_SEC
        frames = []
        for i in range(int(t0) + 1, int(t1) + 2):
            fp = frame_dir / f"frame_{i:06d}.jpg"
            if fp.exists():
                try:
                    frames.append(Image.open(fp).convert("RGB"))
                except Exception:
                    pass
        if not frames:
            return None
        with torch.no_grad():
            inp = proc(images=frames, return_tensors="pt").to(device)
            feats = model.get_image_features(**inp)
            feats = feats.mean(dim=0)
            feats = feats / (feats.norm() + 1e-8)
        return feats.detach().cpu().numpy().astype(np.float32)

    @np.errstate(invalid="ignore")
    def encode_text(texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            inp = proc(text=texts, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=64).to(device)
            feats = model.get_text_features(**inp)
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
        return feats.detach().cpu().numpy().astype(np.float32)

    return encode_chunk, encode_text


def _make_qwen_visual_encoder(model_path: str, device: str, frames_root: Optional[str]):
    """Reuse the agent's own (frozen) Qwen3-VL ViT for retrieval encoding.

    Aligns retrieval similarity with the agent's own visual perception
    — useful when measuring 'what the agent itself could discriminate'
    rather than 'best off-the-shelf retrieval.'
    """
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch

    logger.info(f"Loading Qwen3-VL ViT from {model_path} (frozen)")
    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    visual = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
    if visual is None:
        raise RuntimeError("Could not locate visual tower on Qwen3-VL model.")

    def encode_chunk(video_id: str, chunk_idx: int) -> Optional[np.ndarray]:
        if not frames_root:
            return None
        frame_dir = Path(frames_root) / video_id
        if not frame_dir.exists():
            return None
        t0 = chunk_idx * AGENT_CHUNK_SEC
        t1 = t0 + AGENT_CHUNK_SEC
        imgs = []
        for i in range(int(t0) + 1, int(t1) + 2):
            fp = frame_dir / f"frame_{i:06d}.jpg"
            if fp.exists():
                try:
                    imgs.append(Image.open(fp).convert("RGB"))
                except Exception:
                    pass
        if not imgs:
            return None
        with torch.no_grad():
            inp = proc(images=imgs, return_tensors="pt").to(device)
            pv = inp["pixel_values"]
            feats = visual(pv) if pv.dim() == 4 else visual(pv.unsqueeze(0))
            if feats.dim() == 3:
                feats = feats.mean(dim=1)  # (B, D)
            feats = feats.mean(dim=0)
            feats = feats / (feats.norm() + 1e-8)
        return feats.float().detach().cpu().numpy().astype(np.float32)

    @np.errstate(invalid="ignore")
    def encode_text(texts: List[str]) -> np.ndarray:
        # Pool token embeddings from the LM input table — cheap, aligned
        # with the agent's vocabulary but weaker than SigLIP's text tower.
        with torch.no_grad():
            tok = proc.tokenizer(texts, padding=True, truncation=True,
                                  max_length=64, return_tensors="pt").to(device)
            emb = model.get_input_embeddings()(tok["input_ids"])  # (B, L, D)
            mask = tok["attention_mask"].unsqueeze(-1).float()
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        return pooled.float().detach().cpu().numpy().astype(np.float32)

    return encode_chunk, encode_text


# ---------------------------------------------------------------------------
# Schemes
# ---------------------------------------------------------------------------


def _bm25_scores(query_text: str, archive: List[Dict]) -> np.ndarray:
    try:
        from rank_bm25 import BM25Okapi
        corpus = [a["text"].lower().split() for a in archive]
        bm25 = BM25Okapi(corpus)
        return bm25.get_scores(query_text.lower().split())
    except ImportError:
        qw = set(query_text.lower().split())
        return np.array(
            [len(qw & set(a["text"].lower().split())) for a in archive],
            dtype=np.float32,
        )


def _filter_archive_by_time(archive: List[Dict], time_range) -> List[Dict]:
    if not time_range or not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
        return archive
    try:
        t0, t1 = float(time_range[0]), float(time_range[1])
    except (TypeError, ValueError):
        return archive
    if t0 > t1:
        t0, t1 = t1, t0
    out = []
    for a in archive:
        cs = a["chunk"] * AGENT_CHUNK_SEC
        ce = cs + AGENT_CHUNK_SEC
        if not (ce <= t0 or cs >= t1):
            out.append(a)
    return out or archive


def _topk_by_score(archive: List[Dict], scores: np.ndarray, k: int) -> List[int]:
    if scores.size == 0:
        return []
    order = np.argsort(-scores)[:k]
    return [archive[i]["chunk"] for i in order if scores[i] > -np.inf]


def _scheme_bm25(archive: List[Dict], query: Dict, *, use_time: bool, k: int) -> List[int]:
    qtext = query.get("query", "")
    if not qtext:
        return []
    arc = _filter_archive_by_time(archive, query.get("time_range")) if use_time else archive
    if not arc:
        return []
    scores = _bm25_scores(qtext, arc)
    return _topk_by_score(arc, scores, k)


def _scheme_dense_text(
    archive: List[Dict], query: Dict, *,
    text_cache: EmbeddingCache, video_id: str, k: int,
) -> List[int]:
    qtext = query.get("query", "")
    if not qtext:
        return []
    # Encode query and archive thinks; both go through the same encoder.
    pairs = [(_text_cache_key(video_id, a["chunk"], a["text"]), a["text"]) for a in archive]
    pairs.append(("__query__", qtext))
    embs = text_cache.get_many(pairs)
    qvec = embs["__query__"]
    scores = np.zeros(len(archive), dtype=np.float32)
    for i, a in enumerate(archive):
        v = embs.get(_text_cache_key(video_id, a["chunk"], a["text"]))
        if v is not None:
            scores[i] = float(np.dot(qvec, v))
    return _topk_by_score(archive, scores, k)


def _scheme_hybrid(
    archive: List[Dict], query: Dict, *,
    text_cache: EmbeddingCache, visual_cache: Dict[str, np.ndarray],
    visual_text_encode, video_id: str, k: int, alpha: float,
) -> List[int]:
    qtext = query.get("query", "")
    if not qtext:
        return []
    # BM25 score (text)
    bm25 = _bm25_scores(qtext, archive)
    bm25_n = _minmax(bm25)
    # Visual score: cosine(text_emb_of_query, visual_emb_of_chunk).
    vis_scores = np.zeros(len(archive), dtype=np.float32)
    if visual_cache:
        try:
            qvis = visual_text_encode([qtext])[0]
            for i, a in enumerate(archive):
                key = _visual_cache_key(video_id, a["chunk"])
                v = visual_cache.get(key)
                if v is not None:
                    vis_scores[i] = (float(np.dot(qvis, v)) + 1.0) / 2.0
        except Exception as e:
            logger.debug(f"Hybrid visual score failed: {e}")
    scores = alpha * bm25_n + (1.0 - alpha) * vis_scores
    return _topk_by_score(archive, scores, k)


def _minmax(s: np.ndarray) -> np.ndarray:
    if s.size == 0:
        return s
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-8:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _hit_at_k(retrieved: List[int], gold: List[int], k: int) -> int:
    return int(any(c in gold for c in retrieved[:k]))


def _mrr(retrieved: List[int], gold: List[int]) -> float:
    for rank, c in enumerate(retrieved, 1):
        if c in gold:
            return 1.0 / rank
    return 0.0


def _iou(retrieved: List[int], gold: List[int]) -> float:
    a, b = set(retrieved), set(gold)
    if not (a or b):
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--samples", type=str,
                     help="pass3c JSONL — uses gold <query> from each row's output.")
    src.add_argument("--predictions", type=str,
                     help="Generation JSONL — uses model's predicted <query>.")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Disk cache for text/visual embeddings. Reused across runs.")
    parser.add_argument("--frames_root", type=str, default=None,
                        help="Pre-extracted frames root for visual scheme. "
                             "If unset, hybrid scheme is skipped.")
    parser.add_argument("--schemes", type=str,
                        default="bm25_keyword,bm25_time,dense_text,hybrid")
    parser.add_argument("--top_k", type=str, default="1,3,5")
    parser.add_argument("--max_results", type=int, default=10,
                        help="Top-K depth for retrieval (must be ≥ max(top_k)).")
    parser.add_argument("--text_encoder", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--visual_encoder", type=str, default="siglip",
                        choices=["siglip", "qwen3vl"],
                        help="siglip: contrastive pretrained, best off-the-shelf "
                             "retrieval cosine. qwen3vl: reuse the agent's own "
                             "(frozen) ViT — aligned with agent's perception "
                             "but lower retrieval quality. Use qwen3vl only if "
                             "you specifically want to measure 'what the agent "
                             "itself could discriminate.'")
    parser.add_argument("--siglip_path", type=str,
                        default="google/siglip-base-patch16-224")
    parser.add_argument("--qwen_model_path", type=str, default=None,
                        help="Required when --visual_encoder qwen3vl.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="BM25 weight in hybrid (1.0 = pure BM25, 0.0 = pure visual).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on samples for a quick smoke test.")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    top_ks = [int(k) for k in args.top_k.split(",") if k.strip()]
    max_results = max(args.max_results, max(top_ks))

    src_path = args.samples or args.predictions
    mode = "samples" if args.samples else "predictions"
    logger.info(f"Loading recall samples from {src_path} (mode={mode})")
    samples = _load_samples(src_path, mode)
    if args.limit:
        samples = samples[:args.limit]
    logger.info(f"Loaded {len(samples)} recall samples with valid query+gold")
    if not samples:
        logger.error("Nothing to evaluate — exiting.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build encoders + caches lazily based on requested schemes ──
    text_cache = None
    if "dense_text" in schemes:
        encode_text = _make_text_encoder(args.text_encoder, args.device)
        text_cache = EmbeddingCache(
            os.path.join(args.cache_dir, "text") if args.cache_dir else None,
            encode_text,
        )

    visual_cache: Dict[str, np.ndarray] = {}
    visual_text_encode = None
    if "hybrid" in schemes and args.frames_root:
        if args.visual_encoder == "qwen3vl":
            if not args.qwen_model_path:
                logger.error("--visual_encoder qwen3vl requires --qwen_model_path")
                return
            encode_chunk, visual_text_encode = _make_qwen_visual_encoder(
                args.qwen_model_path, args.device, args.frames_root,
            )
        else:
            encode_chunk, visual_text_encode = _make_visual_encoder(
                args.siglip_path, args.device, args.frames_root,
            )
        # Pre-build visual cache for all (video_id, chunk) pairs in archive.
        vc_dir = Path(args.cache_dir) / "visual" if args.cache_dir else None
        if vc_dir is not None:
            vc_dir.mkdir(parents=True, exist_ok=True)
        seen = set()
        t0 = time.time()
        for s in samples:
            for a in s["archive"]:
                k = _visual_cache_key(s["video_id"], a["chunk"])
                if k in seen:
                    continue
                seen.add(k)
                cached_path = vc_dir / k if vc_dir else None
                if cached_path is not None and cached_path.exists():
                    visual_cache[k] = np.load(cached_path)
                    continue
                vec = encode_chunk(s["video_id"], a["chunk"])
                if vec is not None:
                    visual_cache[k] = vec
                    if cached_path is not None:
                        np.save(cached_path, vec)
        logger.info(
            f"Visual cache built: {len(visual_cache)}/{len(seen)} chunks "
            f"in {time.time() - t0:.1f}s"
        )
    elif "hybrid" in schemes:
        logger.warning("hybrid scheme requested but --frames_root unset; skipping.")
        schemes = [s for s in schemes if s != "hybrid"]

    # ── Run schemes ──
    per_sample_path = out_dir / "recall_audit_per_sample.jsonl"
    summary: Dict[str, Dict] = {s: {"hit": defaultdict(int), "mrr": 0.0,
                                     "iou": 0.0, "n": 0} for s in schemes}

    t0 = time.time()
    with open(per_sample_path, "w") as f_per:
        for i, s in enumerate(samples):
            row = {
                "sample_id": s["sample_id"],
                "video_id": s["video_id"],
                "chunk_idx": s["chunk_idx"],
                "gold": s["gold"],
                "query": s["query"],
                "retrieved": {},
            }
            for scheme in schemes:
                if scheme == "bm25_keyword":
                    r = _scheme_bm25(s["archive"], s["query"], use_time=False, k=max_results)
                elif scheme == "bm25_time":
                    r = _scheme_bm25(s["archive"], s["query"], use_time=True, k=max_results)
                elif scheme == "dense_text":
                    r = _scheme_dense_text(
                        s["archive"], s["query"],
                        text_cache=text_cache, video_id=s["video_id"], k=max_results,
                    )
                elif scheme == "hybrid":
                    r = _scheme_hybrid(
                        s["archive"], s["query"],
                        text_cache=text_cache, visual_cache=visual_cache,
                        visual_text_encode=visual_text_encode,
                        video_id=s["video_id"], k=max_results, alpha=args.alpha,
                    )
                else:
                    continue
                row["retrieved"][scheme] = r
                for k in top_ks:
                    summary[scheme]["hit"][k] += _hit_at_k(r, s["gold"], k)
                summary[scheme]["mrr"] += _mrr(r, s["gold"])
                summary[scheme]["iou"] += _iou(r[:max_results], s["gold"])
                summary[scheme]["n"] += 1
            f_per.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                logger.info(f"  processed {i + 1}/{len(samples)}")

    # ── Aggregate ──
    out_summary = {"n_samples": len(samples), "schemes": {}}
    for scheme in schemes:
        n = max(summary[scheme]["n"], 1)
        sched = {f"hit@{k}": summary[scheme]["hit"][k] / n for k in top_ks}
        sched["mrr"] = summary[scheme]["mrr"] / n
        sched["iou"] = summary[scheme]["iou"] / n
        sched["n"] = summary[scheme]["n"]
        out_summary["schemes"][scheme] = sched

    summary_path = out_dir / "recall_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Done in {time.time() - t0:.1f}s — wrote {summary_path}")
    print("\n=== Recall retrieval audit ===")
    print(f"Samples evaluated: {len(samples)}")
    header = f"{'scheme':<16}" + "".join(f"  hit@{k}" for k in top_ks) + "    MRR     IoU"
    print(header)
    for scheme in schemes:
        s = out_summary["schemes"][scheme]
        cells = [f"  {s[f'hit@{k}']:.3f}" for k in top_ks]
        print(f"{scheme:<16}" + "".join(cells) + f"  {s['mrr']:.3f}  {s['iou']:.3f}")

    # ── Optional wandb push ──
    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"recall_audit_{int(time.time())}",
                config=vars(args),
            )
            for scheme, m in out_summary["schemes"].items():
                for key, val in m.items():
                    run.log({f"recall_audit/{scheme}/{key}": val})
            cols = ["scheme"] + [f"hit@{k}" for k in top_ks] + ["mrr", "iou", "n"]
            tbl = wandb.Table(columns=cols)
            for scheme, m in out_summary["schemes"].items():
                tbl.add_data(scheme, *(m[f"hit@{k}"] for k in top_ks),
                             m["mrr"], m["iou"], m["n"])
            run.log({"recall_audit/table": tbl})
            run.finish()
        except Exception as e:
            logger.warning(f"wandb push failed: {e}")


if __name__ == "__main__":
    main()
