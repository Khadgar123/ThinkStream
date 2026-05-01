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
    compress_event: Optional[Dict] = None,
) -> Dict:
    """Build silent sample. compress_event (when provided) injects
    <compress_trigger range='a-b'/> into user_input so the model can see
    the system event. Mirrors old pass3c compress sample's user_input
    convention so the model learns: trigger appears → observe silently.
    """
    sample_type = "recall_silent" if sample_subtype == "recall+silent" else "silent"
    output_text = build_assistant_content_v12(
        think=think, kind="answer", answer_text="",
    )
    final_user_input = user_input
    inter_chunk = False
    if compress_event:
        tr = compress_event.get("summary", {}).get("time_range", [])
        if isinstance(tr, list) and len(tr) == 2:
            trigger_tag = f"<compress_trigger range='{int(tr[0])}-{int(tr[1])}'/>"
        else:
            trigger_tag = "<compress_trigger/>"
        final_user_input = (trigger_tag +
                             (("\n" + user_input) if user_input else ""))
        inter_chunk = True
    sample = {
        "chunk_idx": chunk_idx,
        "sample_type": sample_type,
        "prompt_type": "SYSTEM_PROMPT",
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": "silent",
        "output": output_text,
        "queries": deepcopy(queries),
        "user_input": final_user_input,
        "recall_result": None,
        "base_role": base_role,
    }
    if inter_chunk:
        # System-event chunks render WITHOUT visual_window in pass5
        sample["v12_inter_chunk"] = True
    return sample


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
                queries_state.append({
                    "question": card.get("question", ""),
                    "ask_time": p.ask_chunk * AGENT_CHUNK_SEC,
                    "answers": [],
                })

        card_id = ds.card_id
        card = cards_map.get(card_id) if card_id else None
        sequence_type = _mech_to_sequence_type(ds.mechanism) if card_id else ""
        # user_input fires only at the ask_chunk for that card
        user_input = ""
        if card_id and ask_chunk_by_card.get(card_id) == c:
            user_input = (card or {}).get("question", "")

        if ds.sample_kind == "patrol":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                base_role="patrol", sample_subtype="patrol",
            ))
        elif ds.sample_kind == "compress_silent":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                base_role="compress_event", sample_subtype="compress_silent",
                compress_event=compress_event_by_chunk.get(c),
            ))
        elif ds.sample_kind == "silent":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                card_id=card_id or "",
                sequence_type=sequence_type, user_input=user_input,
            ))
        elif ds.sample_kind == "recall+silent":
            raw.append(_silent_sample(
                c, _think_for_chunk(rollout, c), queries_state, traj_id,
                card_id=card_id or "", sequence_type=sequence_type,
                base_role="recall_silent", sample_subtype="recall+silent",
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
