"""v12.0 reward components — pure-Python helpers (no transformers/deepspeed).

Kept separate from grpo.py so unit tests can run on CPU-only environments
(matches gdpo_advantage.py separation pattern).

5 components matching V12_REWARD_DICT_KEYS in gdpo_advantage.py:
- outcome           binary correctness
- timing            bucket reward (early=-1, on=+1, late=decay, missed=-0.5)
- format            binary tag/JSON well-formedness
- spam              additive penalty for excess tool calls
- compress_quality  range_iou + text_match for compress turns

Plus _aggregate_v12_advantages: multi-level GRPO advantage mixing
(ReMemR1 ICLR'26 pattern, see V12_ADVANTAGE_MIX_ALPHA in gdpo_advantage.py).

References:
- docs/v12.0_protocol_migration_design.md §5 (reward design)
- /tmp/agent-sft-research/DeepEyesV2/.../vl_agent.py:217 (anti-hacking)
- /tmp/agent-sft-research/ReMemR1/verl/trainer/.../ray_trainer.py:1288-1340
  (multi-level advantage aggregation)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from thinkstream.trainer.gdpo_advantage import (
    V12_DEFAULT_REWARD_WEIGHTS,
    V12_ADVANTAGE_MIX_ALPHA,
)


# ===========================================================================
# Per-chunk reward components
# ===========================================================================


def compute_outcome_reward_v12(
    final_answer: Optional[str],
    gold_answer: str,
    *,
    judge_fn=None,
    answer_form: str = "",
    options: Optional[Sequence[str]] = None,
    correct_option: Any = "",
) -> float:
    """v12 outcome: 1.0 if final_answer matches gold, 0.0 otherwise.

    Form-aware matcher shared by RL, SFT eval, and benchmark eval:
    multiple_choice uses option/correct_option metadata, binary tolerates
    "Yes.", number extracts numerals, short_exact/descriptive use conservative
    normalized substring matching. For descriptive, a caller-supplied judge_fn
    can override the default fuzzy matcher.

    Anti-hacking: if final_answer length > 1000 chars, force 0
    (DeepEyesV2 vl_agent.py:256 same defense vs reward-judge spam).
    """
    if final_answer is None:
        return 0.0
    if len(final_answer) > 1000:
        return 0.0
    fa = final_answer.strip()
    ga = (gold_answer or "").strip()
    if not ga:
        return 0.0
    if judge_fn is not None and (not answer_form or (answer_form or "").lower() == "descriptive"):
        return float(judge_fn(fa, ga))
    from thinkstream.trainer.outcome_match import score_outcome_by_form
    return float(score_outcome_by_form(
        fa,
        options=list(options or []),
        correct_option=correct_option,
        gold_answer=ga,
        answer_form=answer_form,
    ))


def compute_timing_reward_v12(
    answer_chunk: Optional[int],
    visible_start_chunk: Optional[int],
    visible_end_chunk: Optional[int],
    *,
    late_window_chunks: int = 1,
) -> float:
    """v12 timing-bucket reward (FIRST in any released streaming-video work).

    Buckets:
        +1.0  if answer_chunk ∈ [visible_start, visible_end]
        +0.5  if answer_chunk ∈ (visible_end, visible_end + late_window]
                with linear decay over the late window
        -1.0  if answer_chunk < visible_start         ← hallucination
        -0.5  if answer_chunk is None AND visible_start is not None
                                                       ← missed opportunity
         0.0  if visible_start is None (no visibility window — neutral)

    Asymmetric -1.0 vs -0.5: hallucinating an answer before evidence is
    worse than staying silent through a window (silence recoverable;
    wrong assertion poisons memory).
    """
    if visible_start_chunk is None:
        return 0.0
    if answer_chunk is None:
        return -0.5
    if answer_chunk < visible_start_chunk:
        return -1.0
    if visible_end_chunk is None:
        return 1.0
    if answer_chunk <= visible_end_chunk:
        return 1.0
    delay = answer_chunk - visible_end_chunk
    if delay <= late_window_chunks and late_window_chunks > 0:
        return 1.0 - 0.5 * (delay / late_window_chunks)
    return 0.0


def compute_format_reward_v12(assistant_outputs: List[str]) -> float:
    """v12 format: 1.0 if every assistant output parses cleanly, else 0.0.

    Checks each turn for exactly one think block followed by exactly one
    terminal (<answer> XOR <tool_call>), no extra text outside those tags,
    and strict recall/compress tool JSON schema.
    """
    from thinkstream.data.agent_protocol import parse_agent_output_v12
    if not assistant_outputs:
        return 0.0
    for out in assistant_outputs:
        parsed = parse_agent_output_v12(out)
        if parsed.get("format_error"):
            return 0.0
    return 1.0


def compute_spam_score_v12(
    n_recall_calls: int,
    n_compress_calls: int,
    *,
    recall_budget: int = 1,
    compress_budget: int = 1,
) -> float:
    """v12 spam SCORE (positive number; negate via weight).

    Linear additive over budget (NOT super-linear) for stability.
    Weight in V12_DEFAULT_REWARD_WEIGHTS["spam"] is NEGATIVE so this
    score enters the final reward as a subtraction.
    """
    excess_recall = max(0, n_recall_calls - recall_budget)
    excess_compress = max(0, n_compress_calls - compress_budget)
    return 0.5 * excess_recall + 0.3 * excess_compress


# compute_compress_quality_v12 / compute_recall_quality_v12 removed (v12.6).
# Industry consensus (DeepEyesV2 / ReTool / MMSearch-R1 / Practitioner's Guide
# arXiv:2510.01132) — gold-locked tool-quality rewards lock policy to teacher's
# specific compress range / retrieval span, suppressing exploration. Tool
# credit now flows entirely via outcome × GRPO group-norm: a chunk's
# compress / recall is good iff downstream answers in the same trajectory
# are right (compared to other group rollouts on the same video).


# ===========================================================================
# Trajectory-level outcome (multi-question)
# ===========================================================================


def compute_trajectory_outcome_v12(
    rollout_chunk_outputs: List[Dict],     # [{chunk_idx, kind, answer_text, ...}, ...]
    trajectory_questions: List[Dict],       # from emit_trajectories `questions` field
    *,
    answer_window_chunks: int = 5,
    answer_form_judge=None,
) -> Dict[str, float]:
    """v12.4 — multi-question trajectory outcome scoring.

    Replaces single-question outcome (which only credits the first non-empty
    answer in the rollout). For trajectories with multiple cards (max 3 in
    pass3b plan), each question gets independently scored, then averaged.

    Why this matters:
      A trajectory with 3 cards (F5 at chunk 6, M1 at chunk 28, F1 at chunk 41)
      produces 3 chances to answer. Single-question reward only scores the
      first; multi-question averages all 3, giving the policy denser feedback
      and rewarding the trajectory-level memory state that supports later
      questions (e.g., the M1 question at chunk 28 needs memory of chunk
      0-10 events that no longer fit visible_window).

    Args:
      rollout_chunk_outputs: list of parsed model outputs per chunk, ordered
        by chunk_idx. Each dict has at minimum:
          {"chunk_idx": int, "kind": "answer"|"recall"|"compress"|"unknown",
           "answer_text": Optional[str]}
      trajectory_questions: list from emit_trajectories `questions` field,
        each with {card_id, gold_answer, ask_chunks, answer_form, ...}
      answer_window_chunks: how many chunks AFTER ask_chunk count as
        on-time response (matches timing_reward visibility window).
      answer_form_judge: optional callable(model_answer, gold_answer,
        answer_form) → float ∈ [0,1]. If None, uses compute_outcome_reward_v12
        (literal/strict-form rule matcher + fuzzy fallback).

    Returns:
      {
        "outcome":         mean per-question correctness ∈ [0, 1]
        "n_questions":     # of questions in trajectory
        "n_answered":      # of questions where model emitted any answer
        "n_correct":       # of questions answered correctly
        "per_q_outcomes":  list of per-question float scores (for diagnostics)
      }

    Empty trajectory (zero questions): returns outcome=0, n_*=0.
    """
    if not trajectory_questions:
        return {
            "outcome": 0.0, "n_questions": 0, "n_answered": 0,
            "n_correct": 0, "per_q_outcomes": [],
        }

    # Index rollout outputs by chunk_idx for fast lookup
    by_chunk: Dict[int, Dict] = {}
    for out in rollout_chunk_outputs:
        ci = int(out.get("chunk_idx", -1))
        if ci >= 0:
            by_chunk[ci] = out

    per_q_outcomes: List[float] = []
    n_answered = 0
    n_correct = 0

    # v12.13 (2026-05-02) outcome scoring rewrite (P0-1, P0-2):
    #
    #   v12.12 questions[*] schema introduced:
    #     - "ask_chunk": canonical placement.ask_chunk (the question time)
    #     - "answer_chunks": where actual answers should land (≥1)
    #     - "per_emit_answers": [{chunk, value}] per-emit gold for multi_emit
    #
    #   Two-mode scoring:
    #     SINGLE: len(answer_chunks) ≤ 1 (typical 91% of cards, includes
    #             forward / silent_then_response / direct / backward / recall).
    #             Window = [ask_chunk, max(answer_chunks) + slack]. The
    #             window can span up to ~30 chunks for forward (lead 18-32),
    #             not the old hardcoded ask + 5.
    #
    #     MULTI:  len(answer_chunks) > 1 (F5 counting / PN1 narration /
    #             F7 step-progress). EACH expected emit_chunk is scored
    #             with its OWN gold value from per_emit_answers (e.g. F5
    #             gold = "1" at chunk 5, "2" at chunk 10, "3" at chunk 15).
    #
    # `slack` is small (default 2 chunks) so the model gets a brief grace
    # period to emit the answer right after the expected chunk, but not
    # so wide that it can pass by emitting a stale answer many chunks
    # later. answer_window_chunks remains the legacy ask+N fallback when
    # answer_chunks is missing (backward compat with old trajectories).
    SLACK = 2

    event_counts = {
        "gold": 0,
        "matched_correct": 0,
        "matched_wrong": 0,
        "missed": 0,
        "false_positive_early": 0,
        "false_positive_over": 0,
    }

    for q in trajectory_questions:
        answer_chunks: List[int] = []
        for x in q.get("answer_chunks") or []:
            try:
                answer_chunks.append(int(x))
            except (TypeError, ValueError):
                continue
        answer_chunks = sorted(answer_chunks)
        ask_chunks: List[int] = []
        for x in q.get("ask_chunks") or []:
            try:
                ask_chunks.append(int(x))
            except (TypeError, ValueError):
                continue
        ask_chunks = sorted(ask_chunks)
        if not answer_chunks and len(ask_chunks) > 1:
            answer_chunks = list(ask_chunks)
        ask_chunk = q.get("ask_chunk")
        if not isinstance(ask_chunk, int):
            ask_chunk = ask_chunks[0] if ask_chunks else (
                answer_chunks[0] if answer_chunks else -1)
        if ask_chunk < 0 and not answer_chunks:
            per_q_outcomes.append(0.0)
            continue

        per_emit = q.get("per_emit_answers") or []
        gold_default = q.get("gold_answer", "")
        answer_form = q.get("answer_form", "")
        options = q.get("options") or []
        correct_option = q.get("correct_option", "")
        chunk_gold = {
            int(e["chunk"]): str(e.get("value", gold_default))
            for e in per_emit
            if isinstance(e, dict) and e.get("chunk") is not None
        }
        target_chunks = sorted(chunk_gold.keys() or answer_chunks)
        if not target_chunks:
            target_chunks = [ask_chunk + answer_window_chunks]
        gold_events = [
            {"chunk": int(c), "gold": str(chunk_gold.get(int(c), gold_default))}
            for c in target_chunks
        ]
        event_counts["gold"] += len(gold_events)

        scan_start = max(0, int(ask_chunk))
        scan_end = max(int(e["chunk"]) for e in gold_events) + SLACK
        policy_events: List[Dict] = []
        for ci in range(scan_start, scan_end + 1):
            out = by_chunk.get(ci)
            if out and out.get("kind") == "answer" and out.get("answer_text"):
                policy_events.append({"chunk": ci, "text": str(out["answer_text"])})

        used_policy: set[int] = set()
        per_ask_scores: List[float] = []
        for i, gold_event in enumerate(gold_events):
            g_chunk = int(gold_event["chunk"])
            hi = g_chunk + SLACK
            if i + 1 < len(gold_events):
                hi = min(hi, int(gold_events[i + 1]["chunk"]) - 1)
            chosen_idx = None
            for pi, ev in enumerate(policy_events):
                if pi in used_policy:
                    continue
                if g_chunk <= int(ev["chunk"]) <= hi:
                    chosen_idx = pi
                    break
            if chosen_idx is None:
                event_counts["missed"] += 1
                per_ask_scores.append(0.0)
                continue
            used_policy.add(chosen_idx)
            n_answered += 1
            ev = policy_events[chosen_idx]
            if answer_form_judge is not None:
                ask_score = float(answer_form_judge(
                    ev["text"], gold_event["gold"], answer_form,
                ))
            else:
                ask_score = compute_outcome_reward_v12(
                    ev["text"], gold_event["gold"],
                    answer_form=answer_form,
                    options=options,
                    correct_option=correct_option,
                )
            if ask_score >= 1.0:
                n_correct += 1
                event_counts["matched_correct"] += 1
            else:
                event_counts["matched_wrong"] += 1
            per_ask_scores.append(ask_score)

        for pi, ev in enumerate(policy_events):
            if pi in used_policy:
                continue
            ev_chunk = int(ev["chunk"])
            if any(ev_chunk < int(g["chunk"]) for g in gold_events):
                event_counts["false_positive_early"] += 1
            else:
                event_counts["false_positive_over"] += 1

        # Question outcome = mean over its emit chunks (multi: N emits;
        # single: 1 element).
        q_outcome = (
            sum(per_ask_scores) / len(per_ask_scores)
            if per_ask_scores else 0.0
        )
        per_q_outcomes.append(q_outcome)

    mean_outcome = sum(per_q_outcomes) / len(per_q_outcomes)
    return {
        "outcome": mean_outcome,
        "n_questions": len(trajectory_questions),
        "n_answered": n_answered,
        "n_correct": n_correct,
        "per_q_outcomes": per_q_outcomes,
        "event_counts": event_counts,
    }


def compute_per_chunk_silent_quality_v12(
    rollout_chunk_outputs: List[Dict],     # parsed model output per chunk
    gold_action_per_chunk: Dict,            # str(chunk_idx) → gold sample_type
) -> Dict[str, float]:
    """v12.4 — per-chunk silent_quality, averaged over the trajectory.

    Replaces single-point silent_quality (which only scores the trajectory's
    final answer behavior). Now we score EACH chunk's silent/respond decision
    against the placement plan's gold action map.

    For each chunk in the rollout:
      gold says "silent" / model emits empty answer  → +0.3 (correct silence)
      gold says "silent" / model emits answer        → -0.6 (HALLUCINATION)
      gold says "recall_silent" / model recalls      → +0.3 (correct wait check)
      gold says "recall_silent" / model empty-answer →  0.0 (waited but skipped recall)
      gold says "recall_silent" / model answers      → -0.6 (HALLUCINATION)
      gold says "response"/"recall_response" / model emits answer → 0.0
        (correctness handled by trajectory_outcome reward)
      gold says "response"/"recall_response" / model is silent     → -0.6 (MISSED)
      gold says "recall" / model recalls or answers → 0.0, silent → -0.6
      gold says "compress"/"recall_query" — neutral (other rewards apply)

    Returns:
      {
        "silent_quality":   mean score over scored chunks ∈ [-0.6, +0.3]
        "n_chunks_scored":  # chunks with non-neutral gold action
        "n_correct_silent": # correct silence cases
        "n_hallucinate":    # gold-silent but model-talked cases
        "n_missed":         # gold-response but model-silent cases
      }
    """
    n_correct = 0
    n_hallucinate = 0
    n_missed = 0
    score_sum = 0.0
    n_scored = 0

    by_chunk = {int(o.get("chunk_idx", -1)): o for o in rollout_chunk_outputs}

    for ci_str, gold_action in gold_action_per_chunk.items():
        ci = int(ci_str)
        out = by_chunk.get(ci)
        if out is None:
            continue
        kind = out.get("kind", "unknown")
        ans = out.get("answer_text") or ""
        has_answer = (kind == "answer") and bool(ans.strip())

        if gold_action == "silent":
            n_scored += 1
            if not has_answer:
                score_sum += 0.3
                n_correct += 1
            else:
                score_sum += -0.6
                n_hallucinate += 1
        elif gold_action == "recall_silent":
            n_scored += 1
            if kind == "recall":
                score_sum += 0.3
                n_correct += 1
            elif not has_answer:
                score_sum += 0.0
            else:
                score_sum += -0.6
                n_hallucinate += 1
        elif gold_action in ("response", "recall_response"):
            n_scored += 1
            if has_answer:
                score_sum += 0.0  # correctness reward handles this
            else:
                score_sum += -0.6
                n_missed += 1
        elif gold_action == "recall":
            n_scored += 1
            if kind == "recall" or has_answer:
                score_sum += 0.0  # recall/action correctness handled elsewhere
            else:
                score_sum += -0.6
                n_missed += 1
        # else compress / recall_query — neutral, skip

    mean_score = score_sum / n_scored if n_scored > 0 else 0.0
    return {
        "silent_quality": mean_score,
        "n_chunks_scored": n_scored,
        "n_correct_silent": n_correct,
        "n_hallucinate": n_hallucinate,
        "n_missed": n_missed,
    }


def compute_silent_quality_v12(
    final_answer: Optional[str],
    gold_action: str,
    gold_answer: str,
) -> float:
    """v12 silent-quality — closes the two error modes Q3 audit exposed:

    Without this reward, the v12 reward stack lets:
      (a) silent-when-should-respond → only -0.15 (timing × 0.3)
      (b) hallucinate-response-when-should-be-silent → 0.0 (outcome+timing
          masked out when gold_answer empty)

    Score ∈ {-0.6, -0.3, 0.0, +0.3}:
      gold_action=silent (gold_answer empty):
        → +0.3 if final_answer is empty/None  (correct silence)
        → -0.6 if final_answer non-empty       (HALLUCINATION)
      gold_action ∈ {response, recall_response, recall} AND gold_answer present:
        → -0.6 if final_answer empty/None      (MISSED — should have answered)
        →  0.0 otherwise (correctness handled by outcome reward)
      otherwise: 0.0  (compress / recall_query — these have their own rewards)

    Magnitudes match v11's silent_quality (-0.6 worst case) so the swing
    is comparable to the response-correctness reward at weight 0.20.
    """
    has_answer = bool(final_answer and final_answer.strip())
    ga_clean = (gold_action or "").strip().lower()
    gold_present = bool(gold_answer and str(gold_answer).strip())

    if ga_clean == "silent" or (ga_clean in ("", "none") and not gold_present):
        # Should be silent
        return 0.3 if not has_answer else -0.6
    if ga_clean in ("response", "recall_response", "recall") and gold_present:
        # Should respond
        return -0.6 if not has_answer else 0.0
    # compress / recall_query / other — silent_quality is neutral; specific
    # rewards (compress_quality, recall_quality) handle these cases.
    return 0.0


# ===========================================================================
# Multi-level GRPO advantage aggregation (ReMemR1 ICLR'26 pattern)
# ===========================================================================


def aggregate_v12_advantages(
    rewards_per_func: Dict[str, torch.Tensor],
    masks_per_func: Dict[str, torch.Tensor],
    chunk_to_video_uid: torch.Tensor,        # [B] int
    chunk_idx_per_row: torch.Tensor,         # [B] int
    group_size: int,
    *,
    weights: Optional[Dict[str, float]] = None,
    alpha: float = V12_ADVANTAGE_MIX_ALPHA,
) -> torch.Tensor:
    """Multi-level GRPO advantage aggregation.

    final_advantage = α · outcome_advantage + (1-α) · state_advantage

    outcome_advantage: GRPO-norm of `outcome` reward, grouped by video_uid
                       (broadcast: every chunk of one video gets same value)
    state_advantage:   GRPO-norm of weighted sum of (timing + format
                       + compress_q − spam), grouped per (uid, chunk_idx)

    α=0.7 default (ReMemR1's 0.8 is HotpotQA — ThinkStream skews lower
    because per-step signal is denser).
    """
    weights = weights or V12_DEFAULT_REWARD_WEIGHTS

    outcome = rewards_per_func.get("outcome")
    if outcome is None:
        outcome_adv = torch.zeros_like(chunk_to_video_uid, dtype=torch.float)
    else:
        outcome_adv = _per_video_outcome_grpo(outcome, chunk_to_video_uid)

    # v12.3 — minimal reward stack matching DeepEyesV2 (arXiv:2511.05271)
    # philosophy: outcome dominant; tool-specific quality rewards dropped
    # so support_chunks-poor families (CR3/CR6/CR7) aren't masked-out.
    # compress_quality / recall_quality functions remain in this module for
    # legacy callers but are not aggregated into state advantage anymore.
    state_components = [
        "timing", "format", "spam",
        "silent_quality",   # streaming-specific decision quality
    ]
    state_sum = torch.zeros_like(outcome_adv)
    for k in state_components:
        if k not in rewards_per_func:
            continue
        w = weights.get(k, 0.0)
        if w == 0.0:
            continue
        col = rewards_per_func[k] * w
        mask = masks_per_func.get(k, torch.ones_like(col))
        state_sum = state_sum + col * mask

    state_adv = _per_chunk_pos_state_grpo(
        state_sum, chunk_to_video_uid, chunk_idx_per_row,
    )

    return alpha * outcome_adv + (1.0 - alpha) * state_adv


def _per_video_outcome_grpo(
    outcome: torch.Tensor,
    chunk_to_video_uid: torch.Tensor,
) -> torch.Tensor:
    """GRPO-norm outcome per video uid, broadcast to all chunks of that video.

    Within a GRPO group of n rollouts of the same video, subtract the group
    mean. Then broadcast that advantage to every chunk of that rollout.
    """
    out_adv = torch.zeros_like(outcome)
    for uid in chunk_to_video_uid.unique().tolist():
        mask = (chunk_to_video_uid == uid)
        rollout_outcomes = outcome[mask]
        g_mean = rollout_outcomes.mean()
        out_adv[mask] = rollout_outcomes - g_mean
    return out_adv


def _per_chunk_pos_state_grpo(
    state: torch.Tensor,
    chunk_to_video_uid: torch.Tensor,
    chunk_idx_per_row: torch.Tensor,
) -> torch.Tensor:
    """GRPO-norm state reward per (uid, chunk_idx) bucket.

    For each video, for each chunk position, mean-subtract across the
    `group_size` rollouts at that position. Cross-chunk comparison is
    meaningless (different memory states), so we compare ONLY rollouts
    that saw the same chunk_idx of the same video.
    """
    state_adv = torch.zeros_like(state)
    keys = chunk_to_video_uid * 100000 + chunk_idx_per_row.long()
    for k in keys.unique().tolist():
        mask = (keys == k)
        bucket = state[mask]
        if bucket.numel() == 0:
            continue
        state_adv[mask] = bucket - bucket.mean()
    return state_adv
