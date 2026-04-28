"""
Pass 3-D: Sample selection (IFD scoring + facility-location submodular pick).

Pipeline:
  Layer 1 (hard):     OVO-task quota — each task in OVO_TASK_QUOTA gets >= N samples.
  Layer 2 (IFD):      drop samples with IFD outside [IFD_MIN, IFD_MAX].
  Layer 3 (submodular): facility-location greedy over (OVO_task, family) labels.

IFD = ppl(answer | question) / ppl(answer | None).
  Real IFD requires a base student model forward pass. This module provides
  two backends:
    - "heuristic"  (default, no model): proxy from answer length + question length
    - "model"      (set IFD_BACKEND env): real ppl ratio via Qwen3-VL-8B
  The interface is stable; switch backends without changing callers.

Output: selected list of samples written to data/agent_v5/selected/{video_id}.json
"""

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import DATA_ROOT
from .pass3a_cards import FAMILY_TARGETS

logger = logging.getLogger(__name__)

SELECTED_DIR = DATA_ROOT / "selected"

# OVO task → minimum samples (Layer 1 hard floor).
# Tuned for ~3500 final samples post-submodular over 100 videos.
# v9.4: CRR/EPM/ASI/SSR bumped — reasoning families (CR1/CR2/CR4) now
# contribute to these, and we want a higher floor so the model gets enough
# multi-chunk-reasoning training signal to actually do well on those tasks.
OVO_TASK_QUOTA = {
    "OCR": 300, "ATR": 300, "STU": 300, "ACR": 300, "OJR": 300,
    "REC": 250, "FPD": 250, "HLD": 500,    # HLD up: N1 is now MC, more verify yield
    "CRR": 400,                              # CRR up: E2(MC) + CR1 + CR2 contributions
    "EPM": 380,                              # EPM up: C1(MC) + CR1 contributions
    "ASI": 380, "SSR": 380,                  # ASI/SSR up: P1(MC) + CR2 + CR3 contributions
}

# Family → OVO task(s). A family may serve multiple OVO tasks; quota is
# satisfied by counting the sample once per task it covers.
# CRR (clue-reveal reasoning) is co-served by E2 event_watch placements
# (silent before / response after the trigger chunk).
# SSR (sequential step recognition) is co-served by P1 procedure samples
# whose answer_form="binary" ("are we currently doing step X?").
# v9.4: CR1/CR2/CR3/CR4 reasoning families added; mapping picks the OVO task
# whose underlying skill the family most directly trains.
FAMILY_TO_OVO: Dict[str, List[str]] = {
    "F1": ["OCR"],
    "F2": ["ATR"],
    "F3": ["OJR"],
    "F4": ["STU"],
    "E1": ["ACR"],
    "E2": ["EPM", "CRR"],
    "P1": ["ASI", "SSR"],
    "C1": ["EPM"],
    "R1": ["OJR"],
    "S1": ["ATR"],
    "M1": ["ACR"],
    "F5": ["REC"],
    "F6": ["FPD"],
    "F7": ["SSR"],            # v9.5 — step-progress binary multi-probe (primary OVO SSR alignment)
    "N1": ["HLD"],
    "CR1": ["CRR", "EPM"],   # causal why = clue-reveal; effect-after-cause = EPM
    "CR2": ["ASI", "SSR"],   # event ordering = action sequence inference
    "CR3": ["ASI"],          # goal/intent = high-level action sequence understanding
    "CR4": ["OJR", "CRR"],   # compositional AND/OR over observations
    "CR5": ["CRR"],           # v9.5 — clue-delayed descriptive multi-probe (primary OVO CRR alignment)
    "CR6": [],                # v9.5 — STAR-Feasibility, no OVO mapping (still scored by IFD/submodular)
    "CR7": [],                # v9.5 — PerceptionTest object permanence, no OVO mapping
}


def _ovo_tasks_for(family: str) -> List[str]:
    """Resolve family → list of OVO tasks (legacy callers may pass scalar)."""
    v = FAMILY_TO_OVO.get(family, [])
    if isinstance(v, str):
        return [v]
    return list(v)

# Layer 2 thresholds. Calibrated for IFD ∈ [0.5, 8] sweet spot (Cherry-LLM).
# Heuristic backend produces values in roughly the same scale.
IFD_MIN = 0.5
IFD_MAX = 8.0


# ---------------------------------------------------------------------------
# Layer 2: IFD scoring
# ---------------------------------------------------------------------------


def compute_ifd_heuristic(sample: Dict) -> float:
    """Heuristic IFD proxy (no model forward).

    Approximates "instruction-following difficulty" by:
      - longer answers relative to question → higher difficulty
      - rarer canonical_answer values → higher difficulty (information content)
      - short binary "Yes"/"No" with long question → easy
    Output is in IFD-comparable scale [~0.5, ~10].
    """
    metadata = sample.get("metadata", {})
    question = metadata.get("question", sample.get("user_input", ""))
    answer = metadata.get("gold_answer", "")
    answer_form = metadata.get("answer_form", "")

    q_words = max(len(question.split()), 1)
    a_words = max(len(answer.split()), 1)

    # Base difficulty: log-scaled question complexity + answer-info ratio.
    # Calibrated so typical samples land in [0.8, 4.0], matching real IFD scale.
    base = 1.5 + 0.4 * math.log(1 + q_words) + 0.3 * a_words / max(q_words, 1)
    if answer_form == "binary":
        base *= 0.7   # Yes/No is easier
    elif answer_form == "multiple_choice":
        base *= 0.9   # MC has explicit options
    elif answer_form == "number":
        base *= 1.1   # exact number requires retention
    elif answer_form == "short_exact":
        base *= 1.2   # exact phrase requires retention
    elif answer_form == "descriptive":
        base *= 1.4   # descriptive long-form

    # Penalize trivially short questions (likely not visual-grounded).
    if q_words < 4:
        base *= 0.6

    return max(0.1, min(base, 10.0))


def compute_ifd_model(sample: Dict, model, tokenizer) -> float:
    """Real IFD via base student model perplexity.

    IFD = ppl(answer | question) / ppl(answer | None)
        ≈ exp(loss_with_q - loss_without_q)

    Caller is responsible for providing a loaded base model + tokenizer.
    """
    import torch  # local import; module is heuristic-only by default

    metadata = sample.get("metadata", {})
    question = metadata.get("question", "")
    answer = metadata.get("gold_answer", "")
    if not question or not answer:
        return 1.0  # no signal

    def _loss(prefix: str, target: str) -> float:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if not target_ids:
            return 0.0
        ids = torch.tensor([prefix_ids + target_ids], device=model.device)
        labels = ids.clone()
        labels[0, :len(prefix_ids)] = -100
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels)
        return out.loss.item()

    loss_with_q = _loss(f"Question: {question}\nAnswer: ", answer)
    loss_without_q = _loss("", answer)
    return math.exp(loss_with_q - loss_without_q)


def score_sample_ifd(sample: Dict, backend: str = "heuristic", **kwargs) -> float:
    if backend == "model":
        return compute_ifd_model(sample, kwargs["model"], kwargs["tokenizer"])
    return compute_ifd_heuristic(sample)


# ---------------------------------------------------------------------------
# Layer 3: submodular facility-location greedy
# ---------------------------------------------------------------------------


def _label_vector(sample: Dict) -> Tuple[str, ...]:
    """Diversity label vector: (ovo_primary_task, family, sequence_type, answer_form).

    A family may serve multiple OVO tasks (e.g. P1 → [ASI, SSR]); we use the
    first task as the primary key so label_vec is hashable. The Layer-1 quota
    step still credits both tasks via _ovo_tasks_for, so multi-mapping is honored
    where it matters (quota satisfaction).
    """
    metadata = sample.get("metadata", {})
    family = metadata.get("family", sample.get("family", "?"))
    tasks = _ovo_tasks_for(family)
    ovo = tasks[0] if tasks else "?"
    return (
        ovo,
        family,
        sample.get("sequence_type", "?"),
        metadata.get("answer_form", "?"),
    )


def _label_similarity(a: Tuple[str, ...], b: Tuple[str, ...]) -> float:
    """Fraction of matching label dimensions."""
    if not a or not b:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / max(len(a), 1)


def submodular_select(
    samples: List[Dict],
    target_count: int,
    pinned: Optional[Set[int]] = None,
) -> List[int]:
    """Facility-location greedy on label buckets (fast).

    Since _label_similarity only depends on the label tuple, samples sharing
    a tuple are interchangeable — we bucket by label and run greedy over
    buckets. Bucket "weight" = number of samples in it. Per-pick cost drops
    from O(n²) to O(|buckets|²), where |buckets| ≤ ~500 in practice.

    pinned: indices that MUST appear in output (Layer 1 quota).
    Returns list of selected sample indices (sorted).
    """
    n = len(samples)
    if n == 0 or target_count <= 0:
        return []
    if target_count >= n:
        return list(range(n))

    pinned = set(pinned or [])
    label_vecs = [_label_vector(s) for s in samples]

    # Pinned overflow: trim within pinned via the same algorithm.
    if len(pinned) > target_count:
        logger.warning(
            f"  3-D submodular: pinned={len(pinned)} > target={target_count}, "
            f"trimming pinned via submodular sub-selection"
        )
        pinned_list = sorted(pinned)
        sub_samples = [samples[i] for i in pinned_list]
        sub_selected = submodular_select(sub_samples, target_count, pinned=set())
        return sorted([pinned_list[k] for k in sub_selected])

    # Bucket samples by label. Within a bucket, prefer pinned then high IFD.
    buckets: Dict[Tuple[str, ...], List[int]] = {}
    for i, lv in enumerate(label_vecs):
        buckets.setdefault(lv, []).append(i)
    # Sort each bucket: pinned first, then by IFD descending.
    for lv, idxs in buckets.items():
        idxs.sort(key=lambda i: (i not in pinned, -samples[i].get("_ifd", 0.0)))

    bucket_labels = list(buckets.keys())
    bucket_weight = [len(buckets[lv]) for lv in bucket_labels]
    B = len(bucket_labels)

    # Precompute pairwise bucket similarity (B×B; B ≤ ~500).
    sim_mat = [[0.0] * B for _ in range(B)]
    for a in range(B):
        for b in range(B):
            sim_mat[a][b] = _label_similarity(bucket_labels[a], bucket_labels[b])

    # selected_per_bucket[b] = how many samples already picked from bucket b.
    selected_per_bucket = [0] * B

    # Initialize from pinned.
    for i in pinned:
        lv = label_vecs[i]
        b = bucket_labels.index(lv)
        selected_per_bucket[b] += 1

    # best_sim[v] = max similarity from current selection to bucket v.
    best_sim = [0.0] * B
    for b in range(B):
        if selected_per_bucket[b] > 0:
            for v in range(B):
                s = sim_mat[v][b]
                if s > best_sim[v]:
                    best_sim[v] = s

    # Greedy: at each step, pick (bucket, slot) maximizing marginal gain.
    selected: List[int] = list(pinned)
    remaining = target_count - len(selected)
    while remaining > 0:
        best_b = -1
        best_gain = -1.0
        for c in range(B):
            if selected_per_bucket[c] >= bucket_weight[c]:
                continue  # bucket exhausted
            # Marginal gain ≈ sum_v bucket_weight[v] × max(0, sim(v,c) - best_sim[v])
            gain = 0.0
            sm_c = sim_mat
            for v in range(B):
                s = sm_c[v][c]
                if s > best_sim[v]:
                    gain += bucket_weight[v] * (s - best_sim[v])
            # Self-bonus: picking another from the same bucket still has
            # diminishing value — use bucket_weight[c] × (1 - best_sim[c]) as fallback.
            if gain == 0.0 and best_sim[c] < 1.0:
                gain = bucket_weight[c] * (1.0 - best_sim[c]) * 0.01
            if gain > best_gain:
                best_gain = gain
                best_b = c
        if best_b < 0:
            break
        # Take the next sample from bucket best_b
        slot_idx = selected_per_bucket[best_b]
        sample_idx = buckets[bucket_labels[best_b]][slot_idx]
        if sample_idx not in pinned:
            selected.append(sample_idx)
        selected_per_bucket[best_b] += 1
        # Update best_sim: now bucket best_b has a selection
        for v in range(B):
            s = sim_mat[v][best_b]
            if s > best_sim[v]:
                best_sim[v] = s
        remaining = target_count - len(selected)

    return sorted(selected)


# ---------------------------------------------------------------------------
# Layer 1 quota fill
# ---------------------------------------------------------------------------


def _fill_ovo_quota(samples: List[Dict]) -> Set[int]:
    """Pick samples to fill OVO_TASK_QUOTA hard floor.

    Returns indices that MUST be kept (added to submodular `pinned` set).
    Picks per-task: highest IFD score samples first.
    """
    by_task: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for i, s in enumerate(samples):
        family = s.get("metadata", {}).get("family", s.get("family", ""))
        for task in _ovo_tasks_for(family):
            if task in OVO_TASK_QUOTA:
                ifd = s.get("_ifd", 1.0)
                by_task[task].append((ifd, i))

    pinned: Set[int] = set()
    for task, quota in OVO_TASK_QUOTA.items():
        bucket = by_task.get(task, [])
        bucket.sort(key=lambda x: x[0], reverse=True)
        for _, i in bucket[:quota]:
            pinned.add(i)
        logger.info(
            f"  3-D quota: {task} requested={quota} "
            f"available={len(bucket)} pinned={min(quota, len(bucket))}"
        )
    return pinned


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_samples(
    all_samples: List[Dict],
    target_count: int,
    backend: str = "heuristic",
    **kwargs,
) -> List[Dict]:
    """Run all 3 layers. Returns the selected sample subset.

    Layer order is:
      1. Score IFD on every sample (cheap heuristic by default).
      2. Pin OVO-quota using IFD as tiebreaker — runs on ALL samples
         so tail families (F5/F6) whose IFD lies outside [IFD_MIN, IFD_MAX]
         can still be pinned for OVO coverage.
      3. IFD filter on NON-PINNED samples only (drop outliers but keep
         pinned tail-family samples even if their IFD is low).
      4. Submodular facility-location on the surviving pool.

    Earlier versions ran IFD filter (Layer 2) before quota (Layer 1), which
    silently culled tail-family samples before they could satisfy OVO quotas.
    Confirmed via audit: F5(REC) had only 4 cards across 312 videos and
    fell out of the IFD window, so quota slots stayed empty.
    """
    # Layer 1: score IFD on EVERY sample
    for s in all_samples:
        s["_ifd"] = score_sample_ifd(s, backend=backend, **kwargs)

    # Layer 2: pin OVO-quota using IFD as tiebreaker, on full pool
    pinned_global = _fill_ovo_quota(all_samples)
    logger.info(f"  3-D quota: {len(pinned_global)} samples pinned (full pool)")

    # Layer 3: IFD filter on NON-PINNED only — drop outliers but always
    # keep pinned-quota samples regardless of IFD value.
    kept_idx: List[int] = []
    pinned_remap: Dict[int, int] = {}
    for i, s in enumerate(all_samples):
        ifd = s["_ifd"]
        if i in pinned_global or IFD_MIN <= ifd <= IFD_MAX:
            pinned_remap[i] = len(kept_idx) if i in pinned_global else -1
            kept_idx.append(i)
    kept = [all_samples[i] for i in kept_idx]
    n_pinned_kept = sum(1 for v in pinned_remap.values() if v >= 0)
    logger.info(
        f"  3-D IFD filter: {len(kept)}/{len(all_samples)} kept "
        f"(IFD ∈ [{IFD_MIN}, {IFD_MAX}] OR pinned; "
        f"of which {n_pinned_kept} are pinned)"
    )

    if not kept:
        return []

    # Remap pinned indices into the `kept` array for submodular.
    pinned = {pinned_remap[i] for i in pinned_global if pinned_remap.get(i, -1) >= 0}

    # Layer 4: submodular facility-location
    selected_idx = submodular_select(kept, target_count, pinned=pinned)
    selected = [kept[i] for i in selected_idx]
    logger.info(f"  3-D submodular: {len(selected)} final samples")
    return selected


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_selected(samples: List[Dict], path: Optional[Path] = None) -> Path:
    path = path or SELECTED_DIR / "selected.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"  3-D wrote {len(samples)} samples to {path}")
    return path


def cli_main(argv: Optional[List[str]] = None) -> None:
    """Standalone CLI: read all rendered samples, run selection, write output."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default=str(DATA_ROOT / "samples_3c" / "*.json"))
    ap.add_argument("--output", default=str(SELECTED_DIR / "selected.jsonl"))
    ap.add_argument("--target", type=int, default=3500)
    ap.add_argument("--backend", choices=["heuristic", "model"], default="heuristic")
    args = ap.parse_args(argv)

    import glob
    files = sorted(glob.glob(args.input_glob))
    samples: List[Dict] = []
    for fp in files:
        with open(fp) as f:
            samples.extend(json.load(f))
    logger.info(f"Loaded {len(samples)} samples from {len(files)} files")

    selected = select_samples(samples, target_count=args.target, backend=args.backend)
    save_selected(selected, Path(args.output))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cli_main()
