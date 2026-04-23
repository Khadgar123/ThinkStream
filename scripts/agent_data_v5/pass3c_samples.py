"""
Pass 3-C: Trajectory Sample Generation

For each trajectory, walks through key_chunks and generates SFT samples.
Maintains queries_state as it evolves (questions enter, answers accumulate).

397B calls: generate response text and recall queries.

Output: fork_samples/{video_id}.json
"""

import json
import logging
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

from .config import (
    AGENT_CHUNK_SEC,
    SAMPLES_3C_DIR,
    PASS_CONFIG,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response / Query Generation (397B)
# ---------------------------------------------------------------------------


RESPONSE_GEN_PROMPT = """Generate a response for this streaming video agent:
- Question: "{question}"
- Available evidence: {evidence}
- Correct answer: {canonical_answer}
- Answer form: {answer_form}

Requirements:
- Base answer ONLY on the provided evidence
- If answer_form is binary: respond Yes or No
- If answer_form is multiple_choice: respond with the letter (A/B/C/D)
- If answer_form is number: respond with the number
- If answer_form is short_exact: respond in 1-5 words
- If answer_form is descriptive: respond in 40-100 words

Output the response text only:"""

RECALL_QUERY_PROMPT = """Generate a retrieval query for this scenario:
- Question: "{question}"
- Visible memory context: {visible_context}

Generate 3-5 discriminative keywords to locate the relevant past observation.
NO answer values, NO pronouns. Include entity descriptions + action anchors.

Output JSON: {{"query": "keyword1 keyword2 keyword3", "time_range": "{time_range}"}}"""


async def _generate_response(card: Dict, snapshot: Dict, evidence: List[Dict],
                              client, video_id: str, chunk_idx: int) -> Optional[str]:
    """Generate response text via 397B."""
    # Build evidence context from what student can see
    evidence_parts = []
    for seg in snapshot.get("compressed_segments", []):
        evidence_parts.append(f"[{seg['time_range']}] {seg['text'][:100]}")
    for item in snapshot.get("recent_thinks", []):
        evidence_parts.append(f"[{item['time']}] {item.get('text', '')}")
    evidence_text = "\n".join(evidence_parts[-10:]) or "Current visual frames."

    prompt = RESPONSE_GEN_PROMPT.format(
        question=card.get("question", ""),
        evidence=evidence_text,
        canonical_answer=card.get("canonical_answer", ""),
        answer_form=card.get("answer_form", "short_exact"),
    )

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3c", {}).get("max_tokens", 16384),
        temperature=0.3,
        request_id=f"{video_id}_resp_{chunk_idx}",
    )
    if raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')
    return raw or card.get("canonical_answer", "")


async def _generate_recall_query(card: Dict, snapshot: Dict,
                                  client, video_id: str, chunk_idx: int) -> Optional[Dict]:
    """Generate recall query JSON via 397B."""
    visible_parts = []
    for seg in snapshot.get("compressed_segments", []):
        visible_parts.append(f"[{seg['time_range']}] {seg['text'][:80]}")
    for item in snapshot.get("recent_thinks", []):
        visible_parts.append(f"[{item['time']}] {item.get('text', '')}")
    visible_context = "\n".join(visible_parts[-10:]) or "(minimal)"

    all_times = []
    for seg in snapshot.get("compressed_segments", []):
        all_times.extend(seg["time_range"])
    for item in snapshot.get("recent_thinks", []):
        parts = item["time"].split("-")
        all_times.extend(int(p) for p in parts)
    time_range = f"0-{max(all_times)}" if all_times else "0-60"

    prompt = RECALL_QUERY_PROMPT.format(
        question=card.get("question", ""),
        visible_context=visible_context,
        time_range=time_range,
    )

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3c", {}).get("max_tokens", 16384),
        temperature=0.3,
        request_id=f"{video_id}_query_{chunk_idx}",
    )
    if not raw:
        return {"query": card.get("question", "")[:30], "time_range": time_range}

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return {"query": card.get("question", "")[:30], "time_range": time_range}


def _simulate_recall_result(card: Dict, rollout: Dict, ask_chunk: int,
                             noise_type: str = "oracle") -> Dict:
    """Simulate retrieval result from student-accessible content.

    noise_type: oracle(70%) / noisy(20%) / distractor(5%) / failure(5%)
    """
    observations = rollout.get("thinks", [])
    support_chunks = card.get("support_chunks", [])

    if noise_type == "failure":
        return {"source": "failure", "text_content": "No matching results found.",
                "returned_chunks": []}

    if noise_type == "distractor":
        available = [o for o in observations if o["chunk_idx"] not in support_chunks
                     and o["chunk_idx"] < ask_chunk]
        if available:
            pick = random.choice(available)
            return {"source": "distractor",
                    "text_content": f"[{pick['time']}] {pick['think']}",
                    "returned_chunks": [pick["chunk_idx"]]}
        return {"source": "failure", "text_content": "No results.", "returned_chunks": []}

    # oracle or noisy
    obs_lookup = {o["chunk_idx"]: o for o in observations}
    parts = []
    returned = []
    for ec in support_chunks:
        obs = obs_lookup.get(ec)
        if obs and ec < ask_chunk:
            parts.append(f"[{obs['time']}] {obs['think']}")
            returned.append(ec)

    if noise_type == "noisy":
        past_obs = [o for o in observations if o["chunk_idx"] < ask_chunk]
        if past_obs:
            distractor = random.choice(past_obs)
            parts.insert(0, f"[{distractor['time']}] {distractor['think']}")

    content = "\n".join(parts) if parts else "No matching results found."
    return {"source": "historical_frames", "text_content": content,
            "returned_chunks": returned}


# ---------------------------------------------------------------------------
# Sample Construction
# ---------------------------------------------------------------------------


def _make_sample(
    chunk_idx: int, prompt_type: str, action: str,
    think: str, queries: List[Dict],
    snapshot: Dict = None,
    response: str = None, query: Dict = None,
    recall_result: Dict = None, user_input: str = None,
    trajectory_id: str = "", card_id: str = "",
    sequence_type: str = "",
) -> Dict:
    """Build one SFT training sample."""
    # Build output string
    output_parts = [f"<think>{think}</think>"]
    output_parts.append(f"<action>{action}</action>")
    if action == "response" and response:
        output_parts.append(f"<response>{response}</response>")
    elif action == "recall" and query:
        output_parts.append(f'<query>{json.dumps(query, ensure_ascii=False)}</query>')

    # Derive sample_type for phase assignment and verification
    if prompt_type == "POST_RECALL_PROMPT":
        sample_type = "recall_response" if action == "response" else "recall_silent"
    elif action == "recall":
        sample_type = "recall_query"
    elif action == "response":
        sample_type = "response"
    else:
        sample_type = "silent"

    return {
        "chunk_idx": chunk_idx,
        "sample_type": sample_type,
        "prompt_type": prompt_type,
        "trajectory_id": trajectory_id,
        "card_id": card_id,
        "sequence_type": sequence_type,
        "action": action,
        "output": "".join(output_parts),
        "queries": deepcopy(queries),
        "user_input": user_input or "",
        "recall_result": recall_result,
    }


# ---------------------------------------------------------------------------
# Trajectory Sample Generation
# ---------------------------------------------------------------------------


async def generate_trajectory_samples(
    trajectory: Dict,
    cards_map: Dict[str, Dict],
    rollout: Dict,
    evidence: List[Dict],
    client,
    video_id: str,
) -> List[Dict]:
    """Generate all SFT samples for one trajectory.

    Walks key_chunks sequentially, maintains queries_state.
    """
    samples = []
    queries_state = []
    observations = rollout.get("thinks", [])
    obs_by_idx = {o["chunk_idx"]: o for o in observations}
    traj_id = trajectory["trajectory_id"]

    def _get_think(chunk_idx):
        obs = obs_by_idx.get(chunk_idx)
        if obs:
            return obs.get("think", "")
        return ""

    def _get_snapshot(chunk_idx):
        return rollout["snapshots"].get(chunk_idx, rollout["snapshots"].get(str(chunk_idx), {}))

    for placement in trajectory["placements"]:
        card = cards_map.get(placement["card_id"])
        if not card:
            continue

        kc = placement["key_chunks"]
        seq = placement["sequence_type"]
        ask = kc["ask"]
        snapshot = _get_snapshot(ask)
        think = _get_think(ask)

        if seq == "immediate_response":
            resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "response", think, queries_state,
                snapshot=snapshot, response=resp,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            queries_state.append({"question": card["question"], "answers": [resp]})
            # post_silent
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_success":
            # step1: recall
            query_json = await _generate_recall_query(card, snapshot, client, video_id, ask)
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "recall", think, queries_state,
                snapshot=snapshot, query=query_json,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            # step2: post-recall response
            noise = random.random()
            noise_type = "oracle" if noise < 0.7 else "noisy" if noise < 0.9 else "distractor" if noise < 0.95 else "failure"
            recall_result = _simulate_recall_result(card, rollout, ask, noise_type)
            is_failed = noise_type in ("distractor", "failure")
            if is_failed:
                resp = "I could not find enough evidence to answer."
                post_action = "silent"
            else:
                resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
                post_action = "response"
            recall_think = "Recall returned relevant results." if not is_failed else "Recall returned no relevant results."
            samples.append(_make_sample(
                ask, "POST_RECALL_PROMPT", post_action, recall_think, queries_state,
                response=resp if post_action == "response" else None,
                recall_result=recall_result, trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            if post_action == "response":
                queries_state.append({"question": card["question"], "answers": [resp]})
            else:
                queries_state.append({"question": card["question"], "answers": []})
            # post_silent
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_fail_then_found":
            # step1: recall
            query_json = await _generate_recall_query(card, snapshot, client, video_id, ask)
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "recall", think, queries_state,
                snapshot=snapshot, query=query_json,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            # step2: recall fail → silent
            recall_result = _simulate_recall_result(card, rollout, ask, "failure")
            samples.append(_make_sample(
                ask, "POST_RECALL_PROMPT", "silent",
                "Recall returned no relevant results.", queries_state,
                recall_result=recall_result, trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            queries_state.append({"question": card["question"], "answers": []})
            # wait_silent
            for wc in kc.get("wait_silent", []):
                samples.append(_make_sample(
                    wc, "SYSTEM_PROMPT", "silent", _get_think(wc), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            # found_response
            found = kc.get("found_response")
            if found is not None:
                resp = await _generate_response(card, _get_snapshot(found), evidence,
                                                 client, video_id, found)
                samples.append(_make_sample(
                    found, "SYSTEM_PROMPT", "response", _get_think(found), queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append(resp)

        elif seq == "event_watch":
            # ask: silent (event not happened)
            queries_state.append({"question": card["question"], "answers": []})
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "silent", think, queries_state,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            # wait_silent
            for wc in kc.get("wait_silent", []):
                samples.append(_make_sample(
                    wc, "SYSTEM_PROMPT", "silent", _get_think(wc), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            # trigger: response
            trigger = kc.get("trigger")
            if trigger is not None:
                resp = await _generate_response(card, _get_snapshot(trigger), evidence,
                                                 client, video_id, trigger)
                samples.append(_make_sample(
                    trigger, "SYSTEM_PROMPT", "response", _get_think(trigger), queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append(resp)

        elif seq == "multi_response":
            # first response
            resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "response", think, queries_state,
                response=resp, user_input=card["question"],
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))
            queries_state.append({"question": card["question"], "answers": [resp]})
            # no_change_silent
            for sc in kc.get("no_change_silent", []):
                samples.append(_make_sample(
                    sc, "SYSTEM_PROMPT", "silent", _get_think(sc), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            # followup_response
            for fc in kc.get("followup_response", []):
                resp = await _generate_response(card, _get_snapshot(fc), evidence,
                                                 client, video_id, fc)
                samples.append(_make_sample(
                    fc, "SYSTEM_PROMPT", "response", _get_think(fc), queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append(resp)

    return samples


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_samples(video_id: str, samples: List[Dict]):
    SAMPLES_3C_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_3C_DIR / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def load_samples(video_id: str) -> Optional[List[Dict]]:
    path = SAMPLES_3C_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
