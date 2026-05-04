"""Audit pass2 rollouts for stale visual observations.

This is a diagnostic gate, not a semantic verifier.  It looks for the failure
mode where pass2 emits the same or near-identical think for a long time while
pass1 evidence for those chunks is changing.  That pattern usually means the
student-simulation rollout is copying memory instead of grounding the latest
visual window.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "over",
    "under", "left", "right", "center", "middle", "frame", "video", "scene",
    "visible", "text", "white", "black", "still", "same", "latest", "second",
    "continues", "continue", "remain", "remains", "unchanged", "static",
    "during", "throughout", "current",
}
_STATIC_RE = re.compile(
    r"\b(static|unchanged|no changes|no new|completely static|remain[s]? unchanged)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {
        w for w in _norm(text).split()
        if len(w) > 2 and w not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def _evidence_text(chunk: Dict[str, Any]) -> str:
    parts: List[str] = []
    for ent in chunk.get("visible_entities") or []:
        if not isinstance(ent, dict):
            continue
        parts.append(str(ent.get("desc", "")))
        parts.append(str(ent.get("action", "")))
    for fact in chunk.get("atomic_facts") or []:
        if isinstance(fact, dict):
            parts.append(str(fact.get("fact", "")))
        else:
            parts.append(str(fact))
    return " ".join(p for p in parts if p)


def _evidence_drift(
    evidence: Optional[List[Dict[str, Any]]],
    start: int,
    end: int,
) -> Optional[float]:
    """Median drift from the first pass1 evidence chunk in [start, end]."""
    if not evidence or start >= len(evidence):
        return None
    base = _tokens(_evidence_text(evidence[start]))
    vals: List[float] = []
    for i in range(start + 1, min(end + 1, len(evidence))):
        cur = _tokens(_evidence_text(evidence[i]))
        vals.append(1.0 - _jaccard(base, cur))
    return statistics.median(vals) if vals else 0.0


def _think_texts(rollout: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in rollout.get("thinks") or []:
        if isinstance(item, dict):
            out.append(str(item.get("think", "")))
        else:
            out.append(str(item))
    return out


def _exact_runs(texts: List[str]) -> Iterable[Tuple[int, int, int]]:
    norms = [_norm(t) for t in texts]
    if not norms:
        return
    start = 0
    for i in range(1, len(norms) + 1):
        if i == len(norms) or norms[i] != norms[start]:
            yield start, i - 1, i - start
            start = i


def _near_runs(texts: List[str], threshold: float = 0.86) -> Iterable[Tuple[int, int, int]]:
    """Runs whose adjacent think token sets are almost identical."""
    if not texts:
        return
    tok = [_tokens(t) for t in texts]
    start = 0
    for i in range(1, len(texts) + 1):
        keep = i < len(texts) and _jaccard(tok[i - 1], tok[i]) >= threshold
        if not keep:
            yield start, i - 1, i - start
            start = i


def audit_rollouts(
    rollout_map: Dict[str, Dict[str, Any]],
    evidence_map: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    *,
    min_run_chunks: int = 24,
    drift_threshold: float = 0.55,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Return aggregate stale-observation diagnostics for pass2 rollouts."""
    evidence_map = evidence_map or {}
    total_chunks = 0
    hard_runs: List[Dict[str, Any]] = []
    hard_compress: List[Dict[str, Any]] = []
    exact_video_ids = set()
    near_video_ids = set()

    for video_id, rollout in rollout_map.items():
        texts = _think_texts(rollout)
        total_chunks += len(texts)
        evidence = evidence_map.get(video_id)

        for kind, runs in (
            ("exact", _exact_runs(texts)),
            ("near", _near_runs(texts)),
        ):
            for start, end, length in runs:
                if length < min_run_chunks:
                    continue
                drift = _evidence_drift(evidence, start, end)
                if drift is None or drift < drift_threshold:
                    continue
                rec = {
                    "video_id": video_id,
                    "kind": kind,
                    "range": [start, end],
                    "length": length,
                    "pass1_drift": round(float(drift), 3),
                    "think_preview": texts[start][:220],
                }
                hard_runs.append(rec)
                (exact_video_ids if kind == "exact" else near_video_ids).add(video_id)

        for event in rollout.get("compression_events") or []:
            summary = event.get("summary") or {}
            text = str(summary.get("text", ""))
            tr = summary.get("time_range") or []
            if len(tr) < 2 or not _STATIC_RE.search(text):
                continue
            start, end = int(tr[0]), int(tr[1]) - 1
            length = max(0, end - start + 1)
            if length < min_run_chunks:
                continue
            drift = _evidence_drift(evidence, start, end)
            if drift is None or drift < drift_threshold:
                continue
            hard_compress.append({
                "video_id": video_id,
                "trigger_chunk": event.get("trigger_chunk"),
                "range": [start, end],
                "length": length,
                "pass1_drift": round(float(drift), 3),
                "summary_preview": text[:220],
            })

    hard_video_ids = {r["video_id"] for r in hard_runs}
    n_videos = len(rollout_map)
    return {
        "thresholds": {
            "min_run_chunks": min_run_chunks,
            "drift_threshold": drift_threshold,
            "near_jaccard_threshold": 0.86,
        },
        "totals": {
            "videos": n_videos,
            "think_chunks": total_chunks,
            "hard_stale_runs": len(hard_runs),
            "hard_stale_videos": len(hard_video_ids),
            "hard_stale_video_rate": (
                round(len(hard_video_ids) / n_videos, 4) if n_videos else 0.0
            ),
            "hard_exact_videos": len(exact_video_ids),
            "hard_near_videos": len(near_video_ids),
            "hard_static_compress_events": len(hard_compress),
            "hard_static_compress_videos": len({r["video_id"] for r in hard_compress}),
        },
        "top_stale_runs": sorted(
            hard_runs,
            key=lambda r: (r["length"], r["pass1_drift"]),
            reverse=True,
        )[:top_k],
        "top_static_compress": sorted(
            hard_compress,
            key=lambda r: (r["length"], r["pass1_drift"]),
            reverse=True,
        )[:top_k],
    }


def _load_json_dir(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for fp in sorted(path.glob("*.json")):
        if fp.name == "_version":
            continue
        data[fp.stem] = json.loads(fp.read_text())
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--evidence-dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rollouts = _load_json_dir(Path(args.rollout_dir))
    evidence = _load_json_dir(Path(args.evidence_dir)) if args.evidence_dir else {}
    report = audit_rollouts(rollouts, evidence)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
