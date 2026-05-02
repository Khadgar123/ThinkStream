"""Pass 3-C — Trajectory sample rendering (v2 model-agnostic).

For each trajectory's selected placements, walks every chunk in [0, num_chunks)
and emits ONE raw SFT sample per chunk (silent / response / recall+response /
patrol / compress_silent), using v2/design.py as the single source of truth
for gold actions.

Pipeline contract preserved:
  generate_trajectory_samples(trajectory, cards_map, rollout, evidence,
                               client, video_id) -> List[Dict]    (async)
  save_samples / load_samples
"""

from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from thinkstream.data.agent_protocol import build_assistant_content_v12

from .config import AGENT_CHUNK_SEC, PASS_CONFIG, SAMPLES_3C_DIR
from .pass3a_cards import dict_to_card
from .pass3b_placement import _dict_to_placement
from .v2.design import (
    Placement,
    render_video_samples as _design_render,
)
from .v2.llm_prompts import (
    parse_recall_query_response,
    recall_query_prompt,
    response_generation_prompt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — think text + response/recall payload generation
# ---------------------------------------------------------------------------


def _think_for_chunk(rollout: Dict, chunk_idx: int) -> str:
    """Pull think text from question-blind rollout (pass2 output)."""
    for t in rollout.get("thinks", []):
        if int(t.get("chunk_idx", -1)) == chunk_idx:
            return str(t.get("think", "")).strip()
    return ""


def _response_text_for(card: Dict, value: str) -> str:
    """Map gold_emit value → assistant response text (synchronous fast path).

    For MC: emits the option letter (A/B/C/D).
    For binary/number/short_exact: emits the value directly.
    For descriptive: uses canonical_answer text.
    Use _response_text_via_llm when client is provided for richer descriptive text.
    """
    af = card.get("answer_form", "")
    if af in ("multiple_choice", "binary", "number", "short_exact"):
        return value
    return value or card.get("canonical_answer", "")


async def _response_text_via_llm(card: Dict, value: str, client, video_id: str,
                                  chunk_idx: int) -> str:
    """397B-driven response text. Falls back to _response_text_for on failure.

    Only fires for descriptive answers (others have deterministic mapping).
    """
    af = card.get("answer_form", "")
    if af in ("multiple_choice", "binary", "number", "short_exact"):
        return value
    prompt = response_generation_prompt(card, chunk_idx)
    if not prompt:
        return _response_text_for(card, value)
    cfg = PASS_CONFIG.get("pass3c_response", PASS_CONFIG.get("pass3c", {}))
    try:
        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(cfg.get("max_tokens", 4096)),
            temperature=float(cfg.get("temperature", 0.3)),
            request_id=f"{video_id}_3c_resp_{card.get('card_id','?')}_{chunk_idx}",
            enable_thinking=cfg.get("thinking", False),
        )
    except Exception as exc:
        logger.warning(f"[{video_id}] 3c response LLM failed: {exc}")
        return _response_text_for(card, value)
    text = (raw or "").strip().strip('"').strip("'").strip()
    return text or _response_text_for(card, value)


def _recall_query_for(card: Dict, ask_chunk: int) -> Dict:
    """Build recall_query (synchronous fast path).

    Returns card.recall_query if pre-generated, else heuristic.
    """
    if card.get("recall_query"):
        return card["recall_query"]
    grounding = card.get("grounding_frames", [])
    if grounding:
        tr_start = min(grounding) * AGENT_CHUNK_SEC
        tr_end = (max(grounding) + 1) * AGENT_CHUNK_SEC
        time_range = f"{int(tr_start)}-{int(tr_end)}"
    else:
        time_range = ""
    q = card.get("question", "")
    keywords = " ".join(w.lower() for w in q.split() if len(w) > 3)[:80]
    return {"query": keywords, "time_range": time_range}


async def _recall_query_via_llm(card: Dict, client, video_id: str,
                                  chunk_idx: int) -> Dict:
    """397B-driven recall_query. Caches result on card so we don't re-call."""
    if card.get("recall_query"):
        return card["recall_query"]
    prompt = recall_query_prompt(card)
    cfg = PASS_CONFIG.get("pass3c_recall_query", PASS_CONFIG.get("pass3c", {}))
    fallback_tr = ""
    grounding = card.get("grounding_frames") or []
    if grounding:
        fallback_tr = (f"{int(min(grounding) * AGENT_CHUNK_SEC)}-"
                       f"{int((max(grounding) + 1) * AGENT_CHUNK_SEC)}")
    try:
        raw = await client._call_one(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(cfg.get("max_tokens", 4096)),
            temperature=float(cfg.get("temperature", 0.3)),
            request_id=f"{video_id}_3c_rq_{card.get('card_id','?')}_{chunk_idx}",
            enable_thinking=cfg.get("thinking", False),
        )
    except Exception as exc:
        logger.warning(f"[{video_id}] 3c recall_query LLM failed: {exc}")
        return _recall_query_for(card, chunk_idx)
    rq = parse_recall_query_response(raw or "", fallback_time_range=fallback_tr)
    if not rq.get("query"):
        return _recall_query_for(card, chunk_idx)
    card["recall_query"] = rq      # cache for re-use within trajectory
    return rq


def _recall_result_for(card: Dict, rollout: Dict, noise_kind: str) -> Dict:
    """Build a recall_result matching the old pass3c noise vocabulary.

    noise_kind ∈ {oracle, noisy, failure} (set by design.assign_recall_noise).
    """
    grounding = card.get("grounding_frames", [])
    if noise_kind == "failure" or not grounding:
        return {
            "source": "failure",
            "text_content": "No matching results found.",
            "returned_chunks": [],
            "time": "",
        }
    chunks = sorted(grounding)
    if noise_kind == "noisy":
        # Inject a distractor chunk near grounding
        max_c = max(0, int(rollout.get("num_chunks", 1)) - 1)
        chunks = chunks + [min(chunks[-1] + 5, max_c)]
    tr_start = min(chunks) * AGENT_CHUNK_SEC
    tr_end = (max(chunks) + 1) * AGENT_CHUNK_SEC
    return {
        "source": "historical_frames",
        "text_content": (f"Recalled {len(chunks)} frames from "
                         f"t={int(tr_start)}-{int(tr_end)}s."),
        "returned_chunks": chunks,
        "time": f"{int(tr_start)}-{int(tr_end)}",
    }


def _mech_to_sequence_type(mech: str) -> str:
    """Map v2 mechanism → legacy sequence_type for back-compat metadata."""
    return {
        "silent_then_response": "event_watch",
        "direct": "immediate_response",
        "recall_demo": "recall_success",
        "multi_emit": "multi_response",
    }.get(mech, "immediate_response")


# ---------------------------------------------------------------------------
# Sample builders
# ---------------------------------------------------------------------------


def _silent_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    trajectory_id: str, *, card_id: str = "",
    sequence_type: str = "", base_role: str = "active_silent",
    sample_subtype: str = "silent", user_input: str = "",
) -> Dict:
    """Plain silent sample (silent / patrol / recall_silent)."""
    sample_type = "recall_silent" if sample_subtype == "recall+silent" else "silent"
    output_text = build_assistant_content_v12(
        think=think, kind="answer", answer_text="",
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": sample_type,
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "silent",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": None,
        "base_role": base_role,
    }


def _compress_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    trajectory_id: str, compress_event: Dict,
) -> Dict:
    """Compress sample — model emits <tool_call>{"name":"compress",...}</tool_call>.

    This is the SFT supervision signal that teaches the model to output a
    compression tool_call when the system signals memory pressure.

      user_input = <compress_trigger/>                 (boolean signal only)
      output     = <think>...</think><tool_call>{compress with gold summary
                                                  INCLUDING time_range}</tool_call>
      action     = "compress"
      v12_inter_chunk = True (pass5 omits visual_window — system-event chunk)

    The gold summary (time_range + text) comes from rollout's compression_events,
    which were generated by the question-blind pass2 rollout.

    v12.12 (2026-05-02): trigger no longer carries range='a-b'. Range stays
    ONLY in the gold tool_call output — model must learn to derive the range
    from <memory> contents (chunk timestamps + summary boundaries) rather
    than copy it from the system-injected trigger. SFT loss covers every
    token of the assistant turn, including the time_range field, so the
    range-selection policy is distilled from pass2's score_range oracle
    into the model. This unblocks the v12.12 RL upgrade where the trigger
    is removed entirely and the model decides timing AND range.
    """
    summary = compress_event.get("summary", {}) or {}
    tr = summary.get("time_range", [])
    trigger_tag = "<compress_trigger/>"
    if isinstance(tr, list) and len(tr) == 2:
        summary_arg = {"time_range": [int(tr[0]), int(tr[1])],
                       "text": summary.get("text", "")}
    else:
        summary_arg = {"time_range": [], "text": summary.get("text", "")}
    output_text = build_assistant_content_v12(
        think=think, kind="compress", compress_summary=summary_arg,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "compress",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": "",
        "sequence_type": "compress_event",
        "action": "compress",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": trigger_tag,                           # system signal
        "recall_result": None,
        "base_role": "compress_action",
        "v12_inter_chunk": True,
        # ICAE-style aux target: original gold caption / source chunks
        "gold_caption": summary.get("text", ""),
        "gold_compress_chunks": sorted(int(c) for c in
            (compress_event.get("compressed_thinks_chunks") or [])),
    }


def _response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "",
) -> Dict:
    output_text = build_assistant_content_v12(
        think=think, kind="answer", answer_text=response,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "response",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "response",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": None,
    }


def _recall_response_sample(
    chunk_idx: int, think: str, response: str, queries: List[Dict],
    recall_query: Dict, recall_result: Dict,
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "",
) -> Dict:
    """Multi-turn recall sample (v12 protocol).

    sample_type='recall' triggers pass5's two-turn render:
      assistant → tool_call(recall_query)
      tool      → recall_result + recalled_frames
      assistant → final answer
    """
    turn1 = build_assistant_content_v12(
        think=think, kind="recall", recall_query=recall_query,
    )
    turn2_think = "Recalled relevant frames; deriving the answer."
    turn2 = build_assistant_content_v12(
        think=turn2_think, kind="answer", answer_text=response,
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "recall",
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "response",
        "output": turn2,
        "v12_assistant_turn_1": turn1,
        "v12_assistant_turn_2": turn2,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": recall_result,
    }


def _recall_silent_multiturn_sample(
    chunk_idx: int, think: str, queries: List[Dict],
    recall_query: Dict, recall_result: Dict,
    trajectory_id: str, card_id: str, sequence_type: str,
    user_input: str = "",
) -> Dict:
    """Multi-turn recall sample with FAILURE result → empty answer (v12.13).

    Teaches the model: when recall returns no useful content (source=failure),
    DO NOT fabricate an answer — emit empty <answer></answer> (silent).

    SFT rendering (pass5 shape B variant):
      assistant → tool_call(recall_query)        ← turn1: model attempts recall
      tool      → recall_result(source=failure)  ← system event: nothing found
      assistant → think + empty <answer>          ← turn2: model stays silent

    Without this sample shape, design.py:534 emits "recall+silent" intent
    but pass3c renders it as a plain silent (no tool_call, no failure
    recall_result). The model never sees the failure→silent pattern, so
    at inference it may either (a) skip recall entirely, or (b) fabricate
    an answer when recall fails.
    """
    turn1 = build_assistant_content_v12(
        think=think, kind="recall", recall_query=recall_query,
    )
    turn2_think = (
        "Recall returned no matching evidence. Cannot answer — staying silent."
    )
    turn2 = build_assistant_content_v12(
        think=turn2_think, kind="answer", answer_text="",  # ← silent
    )
    return {
        "chunk_idx": chunk_idx,
        "sample_type": "recall",   # same shape B as recall_response
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "silent",        # ← gold action is silent (not response)
        "output": turn2,
        "v12_assistant_turn_1": turn1,
        "v12_assistant_turn_2": turn2,
        "queries": deepcopy(queries),
        "user_input": user_input,
        "recall_result": recall_result,
        "base_role": "recall_silent",
        "_recall_failure": True,    # diagnostic flag for pass3e/audit
    }


# ---------------------------------------------------------------------------
# Public API (matches old pass3c interface)
# ---------------------------------------------------------------------------


async def generate_trajectory_samples(
    trajectory: Dict,
    cards_map: Dict[str, Dict],
    rollout: Dict,
    evidence: List[Dict],
    client=None,                     # unused (no LLM call in v2 path)
    video_id: str = "",
) -> List[Dict]:
    """Render one trajectory's placements into raw per-chunk samples.

    Output samples carry every field render_samples.render_sample() needs
    (chunk_idx, sample_type, action, output, queries, user_input,
    recall_result, sequence_type, card_id, trajectory_id, optional
    v12_assistant_turn_*). render_samples then adds the `input` dict +
    metadata for pass3e/4/5.
    """
    placements_dicts = trajectory.get("placements", [])
    placements = [_dict_to_placement(p) for p in placements_dicts]
    traj_id = trajectory.get("trajectory_id", f"{video_id}_traj0")
    num_chunks = int(rollout.get("num_chunks", 0))

    compress_chunks = [int(e.get("trigger_chunk", -1))
                       for e in rollout.get("compression_events", [])
                       if e.get("trigger_chunk", -1) >= 0]
    # Map trigger_chunk → full event so we can inject <compress_trigger>
    compress_event_by_chunk: Dict[int, Dict] = {
        int(e.get("trigger_chunk", -1)): e
        for e in rollout.get("compression_events", [])
        if e.get("trigger_chunk", -1) >= 0
    }

    # Run design's gold-action pipeline (handles patrol stratification,
    # compress_silent, priority resolution).
    cards_obj = [dict_to_card(c) for c in cards_map.values()]
    placements_by_card: Dict[str, List[Placement]] = {}
    for p in placements:
        placements_by_card.setdefault(p.card_id, []).append(p)
    rng = random.Random(abs(hash(video_id + traj_id)) % (10**6))
    design_samples = _design_render(
        cards_obj, placements_by_card, num_chunks,
        evidence=evidence, rng=rng,
        compression_event_chunks=compress_chunks,
    )

    # Build queries_state evolution chronologically
    queries_state: List[Dict] = []
    queries_idx_by_card: Dict[str, int] = {}
    placements_sorted = sorted(placements, key=lambda p: p.ask_chunk)
    ask_chunk_by_card = {p.card_id: p.ask_chunk for p in placements_sorted}

    raw: List[Dict] = []
    for ds in design_samples:
        c = ds.chunk_idx
        # Add queries that became active by this chunk
        for p in placements_sorted:
            if p.ask_chunk <= c and p.card_id not in queries_idx_by_card:
                card = cards_map.get(p.card_id) or {}
                queries_idx_by_card[p.card_id] = len(queries_state)
                # v12.13 fix (P0-3): include options + answer_form so
                # format_queries_block can render MC choices for pending
                # queries (forward responses fire AFTER ask, with no fresh
                # user_input — model sees only the queries block).
                queries_state.append({
                    "question": card.get("question", ""),
                    "options": list(card.get("options") or []),
                    "answer_form": card.get("answer_form", ""),
                    "ask_time": p.ask_chunk * AGENT_CHUNK_SEC,
                    "answers": [],
                })

        card_id = ds.card_id
        card = cards_map.get(card_id) if card_id else None
        sequence_type = _mech_to_sequence_type(ds.mechanism) if card_id else ""
        # user_input fires only at the ask_chunk for that card.
        #
        # v12.13 (2026-05-02): MC options live ONLY in <queries> block via
        # format_queries_block (queries_state carries options + answer_form;
        # pending MC queries render an "Options: A) ... B) ..." line).
        # Putting options ALSO in user_input was duplicating ~30 tokens
        # per ask (model saw the same A-D list twice — once in <queries>
        # and once in <user_input>). user_input now carries just the
        # question text, parity with non-MC asks.
        user_input = ""
        if card_id and ask_chunk_by_card.get(card_id) == c and card:
            user_input = card.get("question", "")

        if ds.sample_kind == "patrol":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                base_role="patrol", sample_subtype="patrol",
            ))
        elif ds.sample_kind == "compress_silent":
            # SFT must teach the model to OUTPUT compress tool_calls, not
            # observe silently. Render as a compress action sample with
            # gold summary from the rollout's compression event.
            ce = compress_event_by_chunk.get(c)
            if ce:
                raw.append(_compress_sample(
                    c, _think_for_chunk(rollout, c), queries_state,
                    traj_id, ce,
                ))
            else:
                # Defensive fallback if event missing — emit silent
                raw.append(_silent_sample(
                    c, _think_for_chunk(rollout, c), queries_state,
                    traj_id, base_role="compress_event",
                ))
        elif ds.sample_kind == "silent":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                card_id=card_id or "",
                sequence_type=sequence_type, user_input=user_input,
            ))
        elif ds.sample_kind == "recall+silent":
            # v12.13 fix (P0-4): emit a real two-turn shape B sample with
            # tool_call(recall_query) → tool(recall_result, source=failure)
            # → empty <answer>. Was emitting plain silent without any
            # recall machinery, hiding the "failure → silent" supervision
            # signal from the model entirely.
            if client is not None:
                rq = await _recall_query_via_llm(
                    card or {}, client, video_id, c)
            else:
                rq = _recall_query_for(card or {}, c)
            rr = _recall_result_for(card or {}, rollout, "failure")
            raw.append(_recall_silent_multiturn_sample(
                c, _think_for_chunk(rollout, c), queries_state,
                rq, rr, traj_id, card_id or "", sequence_type,
                user_input=user_input,
            ))
        elif ds.sample_kind == "response":
            if client is not None:
                resp = await _response_text_via_llm(
                    card or {}, ds.response_text, client, video_id, c)
            else:
                resp = _response_text_for(card or {}, ds.response_text)
            raw.append(_response_sample(
                c, _think_for_chunk(rollout, c), resp, queries_state,
                traj_id, card_id, sequence_type, user_input=user_input,
            ))
            if card_id in queries_idx_by_card:
                queries_state[queries_idx_by_card[card_id]]["answers"].append({
                    "text": resp, "time": c * AGENT_CHUNK_SEC,
                })
        elif ds.sample_kind == "recall+response":
            if client is not None:
                resp = await _response_text_via_llm(
                    card or {}, ds.response_text, client, video_id, c)
                rq = await _recall_query_via_llm(card or {}, client, video_id, c)
            else:
                resp = _response_text_for(card or {}, ds.response_text)
                rq = _recall_query_for(card or {}, c)
            rr = _recall_result_for(card or {}, rollout,
                                     ds.recall_result_kind or "oracle")
            raw.append(_recall_response_sample(
                c, _think_for_chunk(rollout, c), resp, queries_state,
                rq, rr, traj_id, card_id, sequence_type, user_input=user_input,
            ))
            if card_id in queries_idx_by_card:
                queries_state[queries_idx_by_card[card_id]]["answers"].append({
                    "text": resp, "time": c * AGENT_CHUNK_SEC,
                })

    # v12.12 fix (P0-1): stamp every card-bearing sample with its REAL
    # ask_chunk from placement (not the answer chunk). pass4 currently
    # infers ask_chunk from response sample's chunk_idx, which collapses
    # forward/silent_then_response timing supervision. Now pass4 can read
    # `sample["ask_chunk"]` directly and preserve the silent-then-respond
    # pattern in `questions[*].ask_chunks`.
    #
    # v12.13 fix (P0-2): also stamp `per_emit_answers` from the card's
    # gold_emits. multi_emit cards (F5 counting / PN1 narration / F7 status)
    # have multiple expected answer chunks each with their OWN gold answer
    # — e.g. F5 emits "1" at chunk_5, "2" at chunk_10, "3" at chunk_15.
    # pass4 needs this to score multi-emit per-emit (not just at canonical_ask).
    for s in raw:
        cid = s.get("card_id")
        if not cid:
            continue
        if cid in ask_chunk_by_card:
            s["ask_chunk"] = int(ask_chunk_by_card[cid])
        card = cards_map.get(cid) or {}
        gold_emits = card.get("gold_emits") or []
        if gold_emits:
            s["per_emit_answers"] = [
                {"chunk": int(e["chunk"]), "value": str(e.get("value", ""))}
                for e in gold_emits
                if isinstance(e, dict) and "chunk" in e
            ]

    return raw


def save_samples(video_id: str, samples: List[Dict],
                 output_dir: Path = SAMPLES_3C_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{video_id}.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2)
    )


def load_samples(video_id: str,
                 samples_dir: Path = SAMPLES_3C_DIR) -> Optional[List[Dict]]:
    p = samples_dir / f"{video_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
