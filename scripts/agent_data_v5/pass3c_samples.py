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
{prior_answers_section}
Requirements:
- Base answer ONLY on the provided evidence
- If answer_form is binary: respond Yes or No
- If answer_form is multiple_choice: respond with the letter (A/B/C/D)
- If answer_form is number: respond with the number
- If answer_form is short_exact: respond in 1-5 words
- If answer_form is descriptive: respond in 40-100 words
{dedup_instruction}
Output the response text only:"""

RECALL_QUERY_PROMPT = """Generate a retrieval query for this scenario:
- Question: "{question}"
- Visible memory context: {visible_context}

Generate 3-5 discriminative keywords to locate the relevant past observation.
NO answer values, NO pronouns. Include entity descriptions + action anchors.

Output JSON: {{"query": "keyword1 keyword2 keyword3", "time_range": "{time_range}"}}"""


async def _generate_response(card: Dict, snapshot: Dict, evidence: List[Dict],
                              client, video_id: str, chunk_idx: int,
                              prior_answers: List[str] = None) -> Optional[str]:
    """Generate response text via 397B.

    Args:
        prior_answers: Previously generated answers for the same question
            (multi_response followups). Used to avoid repeating content.
    """
    # Build evidence context from what student can see
    evidence_parts = []
    for seg in snapshot.get("compressed_segments", []):
        evidence_parts.append(f"[{seg['time_range']}] {seg['text'][:100]}")
    for item in snapshot.get("recent_thinks", []):
        evidence_parts.append(f"[{item['time']}] {item.get('text', '')}")
    evidence_text = "\n".join(evidence_parts[-10:]) or "Current visual frames."

    # Build prior-answers section for multi_response dedup
    if prior_answers:
        pa_text = "\n".join(f"  - {a}" for a in prior_answers)
        prior_section = f"- Prior answers already given:\n{pa_text}\n"
        dedup = "- Do NOT repeat information from prior answers. Only describe NEW changes or content."
    else:
        prior_section = ""
        dedup = ""

    prompt = RESPONSE_GEN_PROMPT.format(
        question=card.get("question", ""),
        evidence=evidence_text,
        canonical_answer=card.get("canonical_answer", ""),
        answer_form=card.get("answer_form", "short_exact"),
        prior_answers_section=prior_section,
        dedup_instruction=dedup,
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
        for p in item["time"].split("-"):
            try:
                all_times.append(int(float(p)))
            except (ValueError, TypeError):
                pass
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
    After fork samples, generates selective base samples (silent/compress)
    around question windows and compression events.

    Returns combined fork + base samples, sorted by chunk_idx.
    """
    samples = []
    queries_state = []
    # Track queries_state at each key_chunk boundary for base sample generation
    queries_state_at_chunks: Dict[int, List[Dict]] = {0: []}
    observations = rollout.get("thinks", [])
    obs_by_idx = {o["chunk_idx"]: o for o in observations}
    traj_id = trajectory["trajectory_id"]

    def _get_think(chunk_idx):
        obs = obs_by_idx.get(chunk_idx)
        if obs:
            return obs.get("think", "")
        return ""

    def _get_snapshot(chunk_idx):
        snapshots = rollout["snapshots"]
        return snapshots.get(chunk_idx) or snapshots.get(str(chunk_idx)) or {}

    def _add_query(question, ask_chunk, answer=None, response_chunk=None):
        """Append to queries_state. Each answer carries its own timestamp."""
        entry = {
            "question": question,
            "ask_time": ask_chunk * AGENT_CHUNK_SEC,
            "answers": [],  # list of {"text": ..., "time": ...}
        }
        if answer is not None:
            entry["answers"].append({
                "text": str(answer),
                "time": (response_chunk or ask_chunk) * AGENT_CHUNK_SEC,
            })
        queries_state.append(entry)

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
            _add_query(card["question"], ask, answer=resp, response_chunk=ask)  # answered immediately
            # post_silent
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_success":
            # Register question as pending BEFORE recall (model sees it in queries)
            _add_query(card["question"], ask)
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
            # Update with answer (or leave pending if failed)
            if post_action == "response":
                queries_state[-1]["answers"].append({"text": str(resp), "time": ask * AGENT_CHUNK_SEC})
            # post_silent
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_fail_then_found":
            # Register question as pending BEFORE recall
            _add_query(card["question"], ask)
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
            # wait_silent
            for wc in kc.get("wait_silent", []):
                samples.append(_make_sample(
                    wc, "SYSTEM_PROMPT", "silent", _get_think(wc), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            # found_response + post_silent
            found = kc.get("found_response")
            if found is not None:
                resp = await _generate_response(card, _get_snapshot(found), evidence,
                                                 client, video_id, found)
                samples.append(_make_sample(
                    found, "SYSTEM_PROMPT", "response", _get_think(found), queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append({"text": str(resp), "time": found * AGENT_CHUNK_SEC})
                # post_silent after found_response
                ps = kc.get("post_silent", found + 1)
                samples.append(_make_sample(
                    ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))

        elif seq == "event_watch":
            # ask: silent (event not happened)
            _add_query(card["question"], ask)
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
                queries_state[-1]["answers"].append({"text": str(resp), "time": trigger * AGENT_CHUNK_SEC})

        elif seq == "multi_response":
            # first response
            resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "response", think, queries_state,
                response=resp, user_input=card["question"],
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))
            _add_query(card["question"], ask, answer=resp, response_chunk=ask)
            # no_change_silent
            for sc in kc.get("no_change_silent", []):
                samples.append(_make_sample(
                    sc, "SYSTEM_PROMPT", "silent", _get_think(sc), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            # followup_response — pass prior answers to avoid repetition
            for fc in kc.get("followup_response", []):
                prior = [a["text"] if isinstance(a, dict) else str(a)
                         for a in queries_state[-1]["answers"]]
                resp = await _generate_response(card, _get_snapshot(fc), evidence,
                                                 client, video_id, fc,
                                                 prior_answers=prior)
                samples.append(_make_sample(
                    fc, "SYSTEM_PROMPT", "response", _get_think(fc), queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append({"text": str(resp), "time": fc * AGENT_CHUNK_SEC})

    # Record queries_state at each fork sample's chunk for base interpolation
    for s in samples:
        c = s["chunk_idx"]
        queries_state_at_chunks[c] = deepcopy(s["queries"])

    # Generate selective base samples (silent/compress around question windows)
    base_samples = generate_base_samples(trajectory, rollout, cards_map, queries_state_at_chunks)
    all_samples = samples + base_samples
    all_samples.sort(key=lambda s: s["chunk_idx"])

    return all_samples


# ---------------------------------------------------------------------------
# Base Sample Generation (selective, not every chunk)
# ---------------------------------------------------------------------------

WARMUP_CHUNKS = 3               # first N chunks for cold-start training
QUESTION_WINDOW_BEFORE = 2      # chunks before each key_chunk
QUESTION_WINDOW_AFTER = 3       # chunks after each key_chunk
COMPRESS_WINDOW = 1             # chunks around each compression event
LONG_SILENT_SAMPLE_INTERVAL = 5  # sample every Nth chunk in long silent stretches
EVIDENCE_WINDOW = 2             # chunks around support_chunks (recall evidence)


def _select_base_chunks(
    trajectory: Dict,
    rollout: Dict,
    cards_map: Dict[str, Dict],
) -> List[int]:
    """Select which base chunks to generate samples for.

    Balances training signal density vs completeness. Selected chunks:

    1. Warmup: chunks 0..2 (cold-start, empty memory)
    2. Evidence anchors: support_chunks ± 2 (where recall evidence lives —
       model must learn to observe/retain this info before the question)
    3. Question window: ask-3 .. last_key+5 (context around Q&A)
    4. Compress chain: all chunks between compress trigger-1 and trigger+1,
       preserving the thinks that feed into the compression summary
    5. Long-silent patrol: every 5th chunk in stretches >10 chunks without
       any selected chunk (teaches "Q answered, stay silent" behavior
       with non-empty queries_state)

    Returns sorted, deduplicated chunk indices.
    """
    num_chunks = rollout["num_chunks"]
    selected = set()

    # 1. Warmup
    for c in range(min(WARMUP_CHUNKS, num_chunks)):
        selected.add(c)

    # 2. Evidence anchors (support_chunks — where recall answers come from)
    for placement in trajectory["placements"]:
        card = cards_map.get(placement["card_id"], {})
        for sc in card.get("support_chunks", []):
            for c in range(max(0, sc - EVIDENCE_WINDOW),
                           min(num_chunks, sc + EVIDENCE_WINDOW + 1)):
                selected.add(c)

    # 3. Question windows: ask ± buffer for each key_chunk individually
    #    (NOT min-to-max of all key_chunks — multi_response followups
    #    can span 30 chunks, which would select half the video)
    for placement in trajectory["placements"]:
        kc = placement["key_chunks"]
        for key, val in kc.items():
            chunks_to_window = []
            if isinstance(val, int):
                chunks_to_window.append(val)
            elif isinstance(val, list):
                chunks_to_window.extend(val)
            for anchor in chunks_to_window:
                ws = max(0, anchor - QUESTION_WINDOW_BEFORE)
                we = min(num_chunks - 1, anchor + QUESTION_WINDOW_AFTER)
                for c in range(ws, we + 1):
                    selected.add(c)

    # 4. Compress anchor: trigger ± 1 + first/last 2 of compressed range
    #    Full compressed_thinks_chunks can be 10-15 chunks — selecting all
    #    would bloat base samples. The model only needs to see:
    #    - trigger context (what state triggered compression)
    #    - range boundaries (what was compressed)
    for event in rollout.get("compression_events", []):
        trigger = event.get("trigger_chunk", -1)
        if trigger < 0:
            continue
        for c in range(max(0, trigger - COMPRESS_WINDOW),
                       min(num_chunks, trigger + COMPRESS_WINDOW + 1)):
            selected.add(c)
        # First/last 2 of compressed range (boundary context)
        cc = sorted(event.get("compressed_thinks_chunks", []))
        for c in cc[:2] + cc[-2:]:
            selected.add(c)

    # 5. Long-silent patrol: in gaps >10 without selected chunks,
    #    sample every 5th chunk to teach "stay silent with active queries"
    sorted_sel = sorted(selected)
    patrol_additions = []
    prev = -1
    for s in sorted_sel:
        gap = s - prev
        if gap > 10:
            for c in range(prev + LONG_SILENT_SAMPLE_INTERVAL,
                           s, LONG_SILENT_SAMPLE_INTERVAL):
                patrol_additions.append(c)
        prev = s
    # Also patrol the tail (last selected → end of video)
    if sorted_sel and num_chunks - 1 - sorted_sel[-1] > 10:
        for c in range(sorted_sel[-1] + LONG_SILENT_SAMPLE_INTERVAL,
                       num_chunks, LONG_SILENT_SAMPLE_INTERVAL):
            patrol_additions.append(c)
    selected.update(patrol_additions)

    return sorted(selected)


def generate_base_samples(
    trajectory: Dict,
    rollout: Dict,
    cards_map: Dict[str, Dict],
    queries_state_at_chunks: Dict[int, List[Dict]],
) -> List[Dict]:
    """Generate base (non-fork) samples at selected chunks.

    These are the silent/compress samples between question events.
    Each sample gets the correct queries_state for its position
    in the trajectory timeline.

    Selection covers 5 categories:
    - Warmup (cold-start)
    - Evidence anchors (recall source chunks)
    - Question windows (Q&A context)
    - Compress chains (full compression context)
    - Long-silent patrol (teach "stay silent with active queries")

    Args:
        trajectory: the trajectory dict
        rollout: Pass 2 rollout data
        cards_map: {card_id: card_dict} for evidence anchor lookup
        queries_state_at_chunks: {chunk_idx: queries_state} mapping
            built during fork sample generation, representing the
            queries_state at each key_chunk boundary.
    """
    observations = rollout.get("thinks", [])
    obs_by_idx = {o["chunk_idx"]: o for o in observations}
    traj_id = trajectory["trajectory_id"]

    # Determine which chunks already have fork samples
    fork_chunks = set()
    for placement in trajectory["placements"]:
        kc = placement["key_chunks"]
        for v in kc.values():
            if isinstance(v, int):
                fork_chunks.add(v)
            elif isinstance(v, list):
                fork_chunks.update(v)

    base_chunks = _select_base_chunks(trajectory, rollout, cards_map)
    samples = []

    # Build queries_state interpolation: for any base chunk, use the
    # queries_state from the nearest preceding key_chunk
    sorted_boundaries = sorted(queries_state_at_chunks.keys())

    for chunk_idx in base_chunks:
        if chunk_idx in fork_chunks:
            continue  # already has a fork sample

        # Find the correct queries_state for this chunk
        qs = []
        for boundary in sorted_boundaries:
            if boundary <= chunk_idx:
                qs = queries_state_at_chunks[boundary]
            else:
                break

        # Get think from rollout
        obs = obs_by_idx.get(chunk_idx)
        think = obs.get("think", "") if obs else ""

        # Check if this chunk is a compression event
        is_compress = False
        for event in rollout.get("compression_events", []):
            if event.get("trigger_chunk") == chunk_idx:
                is_compress = True
                break

        if is_compress:
            action = "compress"
            sample_type = "compress"
        else:
            action = "silent"
            sample_type = "silent"

        samples.append(_make_sample(
            chunk_idx=chunk_idx,
            prompt_type="COMPRESS_PROMPT" if is_compress else "SYSTEM_PROMPT",
            action=action,
            think=think,
            queries=qs,
            trajectory_id=traj_id,
            card_id="",
            sequence_type="base",
        ))

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
