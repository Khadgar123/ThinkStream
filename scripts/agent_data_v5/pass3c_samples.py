"""
Pass 3-C: Trajectory Sample Generation

For each trajectory, walks through key_chunks and generates SFT samples.
Maintains queries_state as it evolves (questions enter, answers accumulate).

397B calls: generate response text and recall queries.

Output: fork_samples/{video_id}.json
"""

import asyncio
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


FORK_THINK_PROMPT = """You are a streaming video agent generating a think (incremental visual memory note).

Current visual observation (from base rollout):
{base_think}

Active questions:
{queries_text}

Rewrite the think in 40-60 tokens. Include:
1. What is NEW or CHANGED in the current visual (same as base observation)
2. If any active question relates to what you currently see, note the relevant visual detail

Rules:
- Only observable visual facts
- Describe entities by appearance (clothing, color, material), not by ID
- NO meta-reasoning, NO "I think", NO "the question asks"
- Do NOT answer the question — just note relevant observations
- If no active question relates to current visual: output the base observation unchanged

Output (one paragraph, 40-60 tokens):"""

RECALL_THINK_PROMPT = """You are a streaming video agent that just received recall results.

Question: "{question}"
Recall result: {recall_text}
Recall source: {recall_source}    # one of: historical_frames, distractor, failure

Write a brief analysis (20-40 tokens) of the recall result. Branch by source:
- source="historical_frames": results contain relevant evidence — note what
  was found and how it answers the question.
- source="distractor": results contain content but it does NOT match the
  question (off-topic chunks). Explicitly note that the retrieved evidence
  does not address the question, so you cannot answer from it.
- source="failure": no results returned. Note recall found no matching
  evidence.

NO meta-reasoning ("I think"), NO sounds/smells/emotions.
Focus on factual assessment of the retrieved content.

Output (one paragraph, 20-40 tokens):"""

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

RECALL_QUERY_PROMPT_WITH_RANGE = """Generate a retrieval query for this scenario:
- Question: "{question}"
- Visible memory context: {visible_context}

Generate 3-5 discriminative keywords to locate the relevant past observation.
NO answer values, NO pronouns. Include entity descriptions + action anchors.
The relevant past observation is around time {time_range} seconds.

Output JSON: {{"query": "keyword1 keyword2 keyword3", "time_range": "{time_range}"}}"""

RECALL_QUERY_PROMPT_KEYWORD_ONLY = """Generate a retrieval query for this scenario:
- Question: "{question}"
- Visible memory context: {visible_context}

Generate 3-5 discriminative keywords to locate the relevant past observation.
NO answer values, NO pronouns. Include entity descriptions + action anchors.

Output JSON: {{"query": "keyword1 keyword2 keyword3"}}"""

# Mix ratio of with-time-range vs keyword-only recall queries in SFT.
# Both schemas are valid at inference; the retriever falls back to the
# full archive when time_range is absent. Training on both teaches the
# model to use time_range when it has a confident estimate of when the
# evidence occurred (typically true for the cards in v9.4) and to skip
# it when uncertain.
RECALL_TIME_RANGE_FRACTION = 0.7


# v9.3: forms whose canonical_answer is ALREADY the exact eval-format answer.
# We skip the LLM call for these — using canonical_answer directly guarantees
# OVO-strict-match format (single letter / Yes-No / digit) and saves ~70%
# of pass3c LLM calls. Batch1's MC training labels frequently drifted to
# "A. eggplant" from teacher LLM despite the prompt's letter-only instruction;
# that drift is the dominant reason SFT looks 92% accurate but OVO eval
# would crash on strict letter matching.
_EXACT_FORMS = {"binary", "multiple_choice", "number"}


def _normalize_exact_form_answer(canonical: str, answer_form: str) -> str:
    """Strict canonicalization of canonical_answer for binary/MC/number.

    Card-generation prompts already enforce these formats, but extra defense
    here turns any teacher-side drift (e.g. "A. eggplant", "Yes.") into the
    eval-strict form. Returns "" if the canonical can't be normalized to the
    expected shape — the caller must then skip the sample.
    """
    s = (canonical or "").strip()
    if not s:
        return ""
    if answer_form == "binary":
        head = s.split()[0].rstrip(".,;!?").lower()
        if head in ("yes", "y", "true"):
            return "Yes"
        if head in ("no", "n", "false"):
            return "No"
        return ""
    if answer_form == "multiple_choice":
        # Pick first standalone letter A-D (handles "A", "A.", "A) eggplant",
        # "(A)", "the answer is B").
        m = re.search(r'(?:^|[^A-Za-z])([A-Da-d])(?:[^A-Za-z]|$)', s)
        return m.group(1).upper() if m else ""
    if answer_form == "number":
        # First digit run; reject if it has letters attached ("3rd").
        m = re.search(r'(?<![A-Za-z])(\d+)(?![A-Za-z])', s)
        return m.group(1) if m else ""
    return s


async def _generate_response(card: Dict, snapshot: Dict, evidence: List[Dict],
                              client, video_id: str, chunk_idx: int,
                              prior_answers: List[str] = None,
                              recall_evidence: str = None) -> Optional[str]:
    """Generate response text.

    v9.3 fast path: for binary/multiple_choice/number, skip the LLM and use
    canonical_answer directly (after strict normalization). The teacher LLM's
    rephrased answer adds no value for these forms — the canonical IS the
    answer in eval format. Returns None if normalization fails so caller
    skips the placement (rare; indicates bad pass3a output).

    For short_exact / descriptive: call the teacher LLM as before.

    Args:
        prior_answers: Previously generated answers for the same question
            (multi_response followups). Used to avoid repeating content.
        recall_evidence: When provided (recall_success path), use this as
            the evidence instead of the snapshot. This ensures the response
            is derived from what recall actually returned, not from the
            student's current memory state.
    """
    answer_form = card.get("answer_form", "short_exact")
    canonical = card.get("canonical_answer", "")

    # ─── Fast path: forms where canonical_answer IS the eval-format answer ───
    if answer_form in _EXACT_FORMS:
        normalized = _normalize_exact_form_answer(canonical, answer_form)
        if not normalized:
            # Bad card — skip placement rather than emit a malformed response.
            return None
        return normalized
    # ─── LLM path: short_exact / descriptive ───

    if recall_evidence:
        # Recall path: response must be based on recall result
        evidence_text = recall_evidence
    else:
        # Normal path: build evidence from what student can see
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
        canonical_answer=canonical,
        answer_form=answer_form,
        prior_answers_section=prior_section,
        dedup_instruction=dedup,
    )

    # v11.3: thinking=False per pass3c_response config. Reasons: (a) 70% of
    # response calls skip the LLM entirely via canonical_answer fast-path;
    # (b) the remaining 30% (short_exact / descriptive) are constrained
    # generation where thinking budget went unused empirically. Quality is
    # validated by Pass 4 leakage checks, so format-only failures fail-loud.
    _resp_cfg = PASS_CONFIG.get("pass3c_response", {})
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_resp_cfg.get("max_tokens", 16384),
        temperature=_resp_cfg.get("temperature", 0.3),
        enable_thinking=_resp_cfg.get("thinking", False),
        request_id=f"{video_id}_resp_{chunk_idx}",
    )
    if raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')
    # Return None on failure — caller must skip this sample.
    # Do NOT fallback to canonical_answer (format mismatch risk for descriptive).
    return raw or None


def _compute_recall_time_range(
    card: Dict, snapshot: Dict, slack_sec: int = 4,
) -> str:
    """Pick a tight time_range for a recall query.

    Uses card.support_chunks (the gold evidence chunks) when available —
    that's the only place we have ground-truth knowledge of where the
    answer actually lives. Falls back to "0-max(visible)" when support
    is missing (rare, mostly silent samples).

    `slack_sec` of 4s on each side compensates for: (a) teacher
    label imprecision, (b) the model's natural uncertainty about exact
    boundaries, (c) chunk_sec=2s quantization — a 4s slack ≈ ±2 chunks.
    Without slack the model would learn "predict exact range" which is
    OOD vs how a human user phrases recall.
    """
    support = card.get("support_chunks") or []
    if support:
        t0 = max(0, min(support) * AGENT_CHUNK_SEC - slack_sec)
        t1 = (max(support) + 1) * AGENT_CHUNK_SEC + slack_sec
        return f"{int(t0)}-{int(t1)}"
    # Fallback: full visible history (legacy v9.2 behaviour)
    all_times: List[int] = []
    for seg in snapshot.get("compressed_segments", []):
        all_times.extend(seg["time_range"])
    for item in snapshot.get("recent_thinks", []):
        for p in item["time"].split("-"):
            try:
                all_times.append(int(float(p)))
            except (ValueError, TypeError):
                pass
    return f"0-{max(all_times)}" if all_times else "0-60"


async def _generate_recall_query(card: Dict, snapshot: Dict,
                                  client, video_id: str, chunk_idx: int) -> Optional[Dict]:
    """Generate recall query JSON via 397B.

    Two schemas are produced in mix RECALL_TIME_RANGE_FRACTION=0.7
    (with_time_range) / 0.3 (keyword_only). Both are valid at inference —
    the retriever uses time_range to pre-filter when present and falls
    back to full archive when absent. Training on both teaches the model
    to volunteer time_range only when confident.
    """
    visible_parts = []
    for seg in snapshot.get("compressed_segments", []):
        visible_parts.append(f"[{seg['time_range']}] {seg['text'][:80]}")
    for item in snapshot.get("recent_thinks", []):
        visible_parts.append(f"[{item['time']}] {item.get('text', '')}")
    visible_context = "\n".join(visible_parts[-10:]) or "(minimal)"

    time_range = _compute_recall_time_range(card, snapshot)
    # Deterministic per-(video, chunk) split so re-runs of pass3c with
    # the same video set produce the same schema choice — important for
    # cache stability and reproducibility.
    schema_seed = f"{video_id}_{chunk_idx}_{card.get('card_id', '')}"
    use_time_range = (hash(schema_seed) % 100) < int(RECALL_TIME_RANGE_FRACTION * 100)

    if use_time_range:
        prompt = RECALL_QUERY_PROMPT_WITH_RANGE.format(
            question=card.get("question", ""),
            visible_context=visible_context,
            time_range=time_range,
        )
        fallback = {"query": card.get("question", "")[:30], "time_range": time_range}
    else:
        prompt = RECALL_QUERY_PROMPT_KEYWORD_ONLY.format(
            question=card.get("question", ""),
            visible_context=visible_context,
        )
        fallback = {"query": card.get("question", "")[:30]}

    # v11.3: thinking=False per pass3c_recall_query config. Empirically the
    # query is just 3-5 keywords + optional time_range — keyword extraction
    # under format constraint, not multi-step reasoning. Pass 4's query/
    # result consistency check (pass3c_samples.py:_query_overlaps_chunks)
    # downgrades bad-query samples to "failure" anyway, so unhelpful
    # queries get punished without needing thinking budget.
    _rq_cfg = PASS_CONFIG.get("pass3c_recall_query", {})
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_rq_cfg.get("max_tokens", 16384),
        temperature=_rq_cfg.get("temperature", 0.3),
        enable_thinking=_rq_cfg.get("thinking", False),
        request_id=f"{video_id}_query_{chunk_idx}",
    )
    if not raw:
        return fallback

    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    if parsed is None:
        return fallback

    # Enforce schema: keyword_only path must NOT carry time_range
    # (otherwise both schemas collapse and the keyword_only training
    # signal is lost).
    if not use_time_range:
        parsed.pop("time_range", None)
    return parsed


async def _generate_fork_think(
    base_think: str, queries_state: List[Dict],
    client, video_id: str, chunk_idx: int,
) -> str:
    """Generate query-aware think for fork points via 397B.

    Takes the pass2 base think (question-blind) and rewrites it to
    acknowledge active questions when visually relevant. Each call
    is independent — only needs base_think + queries_state.
    """
    if not queries_state:
        return base_think  # no active questions → base think is fine

    # Format active queries (unanswered only)
    # Note: answers=[] (empty list) means question is pending (asked but not answered).
    # Must check len() explicitly — empty list is falsy in Python.
    pending = [q for q in queries_state if len(q.get("answers", [])) == 0]
    if not pending:
        return base_think  # all answered → no need to rewrite

    queries_text = "\n".join(
        f"- [{q.get('ask_time', '?')}s] {q['question']}"
        for q in pending
    )

    prompt = FORK_THINK_PROMPT.format(
        base_think=base_think,
        queries_text=queries_text,
    )

    # KEEP thinking (pass3c_fork_think config) — fork_think must mention the
    # active question WITHOUT leaking the answer. This requires careful
    # reasoning about what's visible vs what's been observed; without CoT
    # the leakage check at Pass 4 rejects ~40% of samples in early A/B tests.
    _fthink_cfg = PASS_CONFIG.get("pass3c_fork_think", {})
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_fthink_cfg.get("max_tokens", 16384),
        temperature=_fthink_cfg.get("temperature", 0.3),
        enable_thinking=_fthink_cfg.get("thinking", True),
        request_id=f"{video_id}_fthink_{chunk_idx}",
    )
    if raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')
    return raw or base_think


async def _generate_recall_think(
    card: Dict, recall_result: Dict,
    client, video_id: str, chunk_idx: int,
) -> str:
    """Generate analysis think after recall via 397B.

    Replaces the hardcoded "Recall returned relevant results." string.
    Each call is independent.
    """
    recall_text = recall_result.get("text_content", "No results.")
    recall_source = recall_result.get("source", "unknown")

    prompt = RECALL_THINK_PROMPT.format(
        question=card.get("question", ""),
        recall_text=recall_text,
        recall_source=recall_source,
    )

    # v11.3: thinking=False per pass3c_recall_think config. The task is
    # 3-way templating: recall_source ∈ {historical_frames, distractor,
    # failure} maps to one of three rephrasing patterns. Empirically a
    # branching task, not a reasoning task — thinking budget went unused.
    _rt_cfg = PASS_CONFIG.get("pass3c_recall_think", {})
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_rt_cfg.get("max_tokens", 16384),
        temperature=_rt_cfg.get("temperature", 0.3),
        enable_thinking=_rt_cfg.get("thinking", False),
        request_id=f"{video_id}_rthink_{chunk_idx}",
    )
    if raw:
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().strip('"')
    return raw or f"Recall {'returned results' if recall_source != 'failure' else 'found no matching evidence'}."


_STOP_WORDS_RECALL = frozenset({
    "the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
    "and", "or", "it", "are", "were", "be", "been", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "this", "that", "there", "here",
    "not", "but", "if", "so", "what", "which", "who", "how", "when",
    "where", "many", "much", "any", "some", "for", "from", "by", "as",
})


def _extract_query_keywords(query_json: Optional[Dict]) -> set:
    """Pull lowercased content tokens from the query string of a recall query JSON."""
    if not query_json:
        return set()
    q = (query_json.get("query") or "") if isinstance(query_json, dict) else str(query_json)
    tokens = re.findall(r'\b[a-zA-Z0-9]{2,}\b', q.lower())
    return {t for t in tokens if t not in _STOP_WORDS_RECALL}


def _query_overlaps_chunks(
    query_keywords: set, returned_chunks: List[int], rollout: Dict,
    threshold: int = 1,
) -> bool:
    """Does the query share at least `threshold` content keywords with the
    text of the returned chunks?  v9.4 — this validates that the LLM-generated
    query is actually consistent with the recall result we hand back; without
    it the agent learns "any query string works", because the simulator
    returns oracle chunks regardless of query quality.
    """
    if not query_keywords or not returned_chunks:
        return False
    obs_lookup = {o.get("chunk_idx"): o for o in rollout.get("thinks", [])}
    chunk_text = " ".join(
        (obs_lookup.get(c, {}) or {}).get("think", "")
        for c in returned_chunks
    ).lower()
    chunk_tokens = set(re.findall(r'\b[a-zA-Z0-9]{2,}\b', chunk_text))
    return len(query_keywords & chunk_tokens) >= threshold


def _simulate_recall_result(card: Dict, rollout: Dict, ask_chunk: int,
                             noise_type: str = "oracle",
                             query_json: Optional[Dict] = None) -> Dict:
    """Simulate retrieval result from student-accessible content.

    noise_type: oracle(70%) / noisy(20%) / distractor(5%) / failure(5%)

    v9.4 — when query_json is provided, we validate that the query keywords
    actually overlap with the returned chunks' evidence. If the LLM-generated
    query is bogus (no overlap with the gold chunks it should be retrieving),
    we downgrade the noise type to "failure" so the agent learns:
      good query → relevant evidence
      bad query  → no result
    Without this check, the simulator returns oracle chunks regardless of
    query quality, teaching "any query works" — a training-inference mismatch.
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

    # v9.4 — query/result consistency check. If a query was supplied and it
    # has zero content-token overlap with the chunks we'd return, downgrade
    # to failure: this is a "bad query" signal the agent should learn from.
    if query_json is not None and returned:
        kw = _extract_query_keywords(query_json)
        if kw and not _query_overlaps_chunks(kw, returned, rollout, threshold=1):
            return {"source": "failure",
                    "text_content": "No matching results found.",
                    "returned_chunks": []}

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
    elif action == "compress" and snapshot:
        # Compress action must include <summary> tag from rollout compression event
        compress_event = snapshot.get("_compress_event")
        if compress_event:
            summary = compress_event.get("summary", {})
            output_parts.append(f'<summary>{json.dumps(summary, ensure_ascii=False)}</summary>')

    # Derive sample_type for phase assignment and verification
    if prompt_type == "POST_RECALL_PROMPT":
        sample_type = "recall_response" if action == "response" else "recall_silent"
    elif action == "recall":
        sample_type = "recall_query"
    elif action == "response":
        sample_type = "response"
    elif action == "compress":
        sample_type = "compress"
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
        base_think = _get_think(ask)

        if seq == "immediate_response":
            # v9.1: skip fork_think for immediate_response. The model answers
            # in the same chunk, so query-aware think rewriting is wasted —
            # the response itself proves the question was processed. Saves
            # ~30% of fork_think 397B calls. Keep base_think (visual obs).
            think = base_think
            resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
            if resp is None:
                logger.warning(f"  [{video_id}] response gen failed at chunk {ask}, skipping placement")
                continue
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "response", think, queries_state,
                snapshot=snapshot, response=resp,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            _add_query(card["question"], ask, answer=resp, response_chunk=ask)
            # post_silent: base think OK (question already answered)
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_success":
            _add_query(card["question"], ask)
            # v9.4 — query→result coupling. Generate the recall query FIRST
            # (one synchronous LLM call), then simulate the result with the
            # query in hand so _simulate_recall_result can downgrade to
            # "failure" when the query is bogus. This costs one
            # serialization point (~0.5 s) but eliminates the v9.3 bug
            # where any LLM-generated query string returned oracle chunks,
            # teaching the agent that query content doesn't matter.
            query_json = await _generate_recall_query(
                card, snapshot, client, video_id, ask)

            noise = random.random()
            noise_type = "oracle" if noise < 0.7 else "noisy" if noise < 0.9 else "distractor" if noise < 0.95 else "failure"
            recall_result = _simulate_recall_result(
                card, rollout, ask, noise_type, query_json=query_json)
            # Source may have been downgraded to "failure" by the consistency
            # check, so re-derive is_failed from the actual result.
            is_failed = recall_result.get("source") in ("distractor", "failure")

            tasks = [
                _generate_fork_think(base_think, queries_state, client, video_id, ask),
                _generate_recall_think(card, recall_result, client, video_id, ask),
            ]
            if not is_failed:
                tasks.append(_generate_response(
                    card, snapshot, evidence, client, video_id, ask,
                    recall_evidence=recall_result.get("text_content", "")))
            results = await asyncio.gather(*tasks)
            think = results[0]
            recall_think = results[1]
            if is_failed:
                resp = "I could not find enough evidence to answer."
                post_action = "silent"
            else:
                resp = results[2]
                if resp is None:
                    resp = "I could not find enough evidence to answer."
                    post_action = "silent"
                else:
                    post_action = "response"

            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "recall", think, queries_state,
                snapshot=snapshot, query=query_json,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            samples.append(_make_sample(
                ask, "POST_RECALL_PROMPT", post_action, recall_think, queries_state,
                response=resp if post_action == "response" else None,
                recall_result=recall_result, trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            if post_action == "response":
                queries_state[-1]["answers"].append({"text": str(resp), "time": ask * AGENT_CHUNK_SEC})
            ps = kc.get("post_silent", ask + 1)
            samples.append(_make_sample(
                ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))

        elif seq == "recall_fail_then_found":
            _add_query(card["question"], ask)
            # v9.1: queries_state stays unchanged until found_response.
            # Pre-compute recall_result synchronously, then gather all
            # independent LLM calls in two batches (before/after found).
            recall_result = _simulate_recall_result(card, rollout, ask, "failure")
            wait_chunks = kc.get("wait_silent", [])
            found = kc.get("found_response")

            # Batch 1: ask-time think + recall_query + recall_think + wait_silent thinks
            #          + found_think + found_response (all independent, queries_state stable).
            tasks = [
                _generate_fork_think(base_think, queries_state, client, video_id, ask),
                _generate_recall_query(card, snapshot, client, video_id, ask),
                _generate_recall_think(card, recall_result, client, video_id, ask),
            ]
            tasks += [
                _generate_fork_think(_get_think(wc), queries_state, client, video_id, wc)
                for wc in wait_chunks
            ]
            if found is not None:
                tasks.append(_generate_fork_think(
                    _get_think(found), queries_state, client, video_id, found))
                tasks.append(_generate_response(
                    card, _get_snapshot(found), evidence, client, video_id, found))
            results = await asyncio.gather(*tasks)

            think = results[0]
            query_json = results[1]
            recall_think = results[2]
            wait_thinks = results[3:3 + len(wait_chunks)]

            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "recall", think, queries_state,
                snapshot=snapshot, query=query_json,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            samples.append(_make_sample(
                ask, "POST_RECALL_PROMPT", "silent", recall_think, queries_state,
                recall_result=recall_result, trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            for wc, wc_think in zip(wait_chunks, wait_thinks):
                samples.append(_make_sample(
                    wc, "SYSTEM_PROMPT", "silent", wc_think, queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            if found is not None:
                found_think = results[3 + len(wait_chunks)]
                resp = results[4 + len(wait_chunks)]
                if resp is None:
                    logger.warning(f"  [{video_id}] found_response gen failed at chunk {found}")
                    continue
                samples.append(_make_sample(
                    found, "SYSTEM_PROMPT", "response", found_think, queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append({"text": str(resp), "time": found * AGENT_CHUNK_SEC})
                ps = kc.get("post_silent", found + 1)
                samples.append(_make_sample(
                    ps, "SYSTEM_PROMPT", "silent", _get_think(ps), queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))

        elif seq == "event_watch":
            _add_query(card["question"], ask)
            # v9.1: queries_state is unchanged through ask + wait_silent + trigger
            # (no _add_query, no answer append) — all thinks are independent of
            # each other, so gather them all in one batch. trigger response also
            # independent of trigger think → include in same batch.
            wait_chunks = kc.get("wait_silent", [])
            trigger = kc.get("trigger")
            think_tasks = [
                _generate_fork_think(base_think, queries_state, client, video_id, ask)
            ]
            think_tasks += [
                _generate_fork_think(_get_think(wc), queries_state, client, video_id, wc)
                for wc in wait_chunks
            ]
            if trigger is not None:
                think_tasks.append(_generate_fork_think(
                    _get_think(trigger), queries_state, client, video_id, trigger))
                think_tasks.append(_generate_response(
                    card, _get_snapshot(trigger), evidence, client, video_id, trigger))
            results = await asyncio.gather(*think_tasks)

            # Unpack in order: ask_think, *wait_thinks, [trigger_think, resp]
            ask_think = results[0]
            wait_thinks = results[1:1 + len(wait_chunks)]
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "silent", ask_think, queries_state,
                user_input=card["question"], trajectory_id=traj_id,
                card_id=card["card_id"], sequence_type=seq,
            ))
            for wc, wc_think in zip(wait_chunks, wait_thinks):
                samples.append(_make_sample(
                    wc, "SYSTEM_PROMPT", "silent", wc_think, queries_state,
                    trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                ))
            if trigger is not None:
                trigger_think = results[1 + len(wait_chunks)]
                resp = results[2 + len(wait_chunks)]
                if resp is None:
                    logger.warning(f"  [{video_id}] trigger response gen failed at chunk {trigger}")
                    continue
                samples.append(_make_sample(
                    trigger, "SYSTEM_PROMPT", "response", trigger_think, queries_state,
                    response=resp, trajectory_id=traj_id,
                    card_id=card["card_id"], sequence_type=seq,
                ))
                queries_state[-1]["answers"].append({"text": str(resp), "time": trigger * AGENT_CHUNK_SEC})

        elif seq == "multi_response":
            # v9.1: first response at ask_chunk — skip fork_think (same logic
            # as immediate_response). Followup chunks below STILL get fork_think
            # because the question stays active across silent gaps.
            think = base_think
            resp = await _generate_response(card, snapshot, evidence, client, video_id, ask)
            if resp is None:
                logger.warning(f"  [{video_id}] multi_response gen failed at chunk {ask}, skipping")
                continue
            samples.append(_make_sample(
                ask, "SYSTEM_PROMPT", "response", think, queries_state,
                response=resp, user_input=card["question"],
                trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
            ))
            _add_query(card["question"], ask, answer=resp, response_chunk=ask)
            # v9.1: no_change_silent thinks all see the same queries_state
            # (no mutation in loop) → gather them.
            sc_chunks = kc.get("no_change_silent", [])
            if sc_chunks:
                sc_thinks = await asyncio.gather(*[
                    _generate_fork_think(_get_think(sc), queries_state, client, video_id, sc)
                    for sc in sc_chunks
                ])
                for sc, sc_think in zip(sc_chunks, sc_thinks):
                    samples.append(_make_sample(
                        sc, "SYSTEM_PROMPT", "silent", sc_think, queries_state,
                        trajectory_id=traj_id, card_id=card["card_id"], sequence_type=seq,
                    ))
            # followup_response: queries_state mutates each iteration via answer
            # append, so cross-iteration parallel is unsafe. Within-iteration
            # gather of (fc_think, resp) is safe — both use the same queries_state.
            #
            # v9.5: progressive_answers — when placement carries a
            # {chunk_idx: answer_str} map (F5/F7/CR5 multi_probe families),
            # the per-probe gold answer overrides card.canonical_answer.
            # F5 → cumulative count "1"/"2"/"3"; F7 → "No"/"Yes" flip;
            # CR5 → silent before clue / descriptive after. Without this
            # override every probe would reuse the same final answer
            # (the bug that made OVO REC/SSR/CRR untrainable).
            progressive = (kc.get("progressive_answers") or {}) if isinstance(kc, dict) else {}
            for fc in kc.get("followup_response", []):
                prior = [a["text"] if isinstance(a, dict) else str(a)
                         for a in queries_state[-1]["answers"]]
                # If the placement specified a per-chunk gold answer, use
                # it directly (no LLM call) — these are deterministic.
                progressive_answer = progressive.get(fc)
                if progressive_answer is None:
                    progressive_answer = progressive.get(str(fc))
                if progressive_answer is not None:
                    fc_think = await _generate_fork_think(
                        _get_think(fc), queries_state, client, video_id, fc,
                    )
                    resp = str(progressive_answer)
                else:
                    fc_think, resp = await asyncio.gather(
                        _generate_fork_think(_get_think(fc), queries_state, client, video_id, fc),
                        _generate_response(card, _get_snapshot(fc), evidence,
                                           client, video_id, fc, prior_answers=prior),
                    )
                if resp is None:
                    logger.warning(f"  [{video_id}] followup response gen failed at chunk {fc}")
                    continue
                samples.append(_make_sample(
                    fc, "SYSTEM_PROMPT", "response", fc_think, queries_state,
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

    # v9: enrich compress samples with gold_caption (ICAE auxiliary loss target).
    # Done here (post-render) so we don't have to thread evidence into helpers.
    _enrich_compress_with_gold_caption(all_samples, rollout, evidence)

    return all_samples


def _enrich_compress_with_gold_caption(
    samples: List[Dict], rollout: Dict, evidence: List[Dict],
) -> None:
    """Attach gold_caption to compress samples (ICAE-style aux target).

    The gold_caption is a concatenation of high-confidence atomic_facts +
    state_changes from the chunks being compressed. The trainer can use it
    as a reconstruction target to prevent summary degeneration into vague
    "the person continues" platitudes.

    Mutates samples in place: adds `gold_caption: str` to compress samples.
    """
    if not evidence:
        return
    ev_by_idx = {cap.get("chunk_idx", 0): cap for cap in evidence}
    events_by_trigger = {
        e.get("trigger_chunk"): e
        for e in rollout.get("compression_events", [])
    }

    for s in samples:
        if s.get("sample_type") != "compress":
            continue
        event = events_by_trigger.get(s["chunk_idx"])
        if not event:
            continue
        compressed_chunks = event.get("compressed_thinks_chunks", [])
        facts: List[str] = []
        for c in compressed_chunks:
            cap = ev_by_idx.get(c, {})
            for f in cap.get("atomic_facts", []):
                if f.get("confidence", 0) >= 0.75:
                    facts.append(f["fact"])
            for sc in cap.get("state_changes", []):
                facts.append(str(sc))
        # Cap at 280 tokens (~12 facts) to bound aux loss compute
        gold_caption = " | ".join(facts[:12]).strip()
        if gold_caption:
            s["gold_caption"] = gold_caption


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
) -> Dict[int, str]:
    """Select which base chunks to generate samples for, with role labels.

    Returns {chunk_idx: base_role} where base_role is one of:
      evidence_anchor, compress_boundary, question_window,
      warmup, patrol

    The role determines training loss weight via _get_sample_weight:
    - evidence_anchor: HIGH weight — model must learn to observe/retain
      facts that will be needed for future recall
    - compress_boundary: MEDIUM weight — critical for compression quality
    - question_window: MEDIUM weight — context around Q&A events
    - warmup: LOW weight — cold-start, empty memory
    - patrol: LOW weight — long-silent stretches
    """
    num_chunks = rollout["num_chunks"]
    # Use dict to track role; later roles override earlier (higher priority wins)
    chunk_role: Dict[int, str] = {}

    # 1. Warmup (lowest priority)
    for c in range(min(WARMUP_CHUNKS, num_chunks)):
        chunk_role[c] = "warmup"

    # 5. Long-silent patrol (low priority, computed early so higher-priority overrides)
    # We compute patrol positions first, then let evidence/compress/question override
    all_selected = set(chunk_role.keys())

    # Pre-compute evidence + question + compress chunks for patrol gap detection
    evidence_chunks = set()
    for placement in trajectory["placements"]:
        card = cards_map.get(placement["card_id"], {})
        for sc in card.get("support_chunks", []):
            for c in range(max(0, sc - EVIDENCE_WINDOW),
                           min(num_chunks, sc + EVIDENCE_WINDOW + 1)):
                evidence_chunks.add(c)

    question_chunks = set()
    for placement in trajectory["placements"]:
        kc = placement["key_chunks"]
        for key, val in kc.items():
            anchors = [val] if isinstance(val, int) else (val if isinstance(val, list) else [])
            for anchor in anchors:
                ws = max(0, anchor - QUESTION_WINDOW_BEFORE)
                we = min(num_chunks - 1, anchor + QUESTION_WINDOW_AFTER)
                for c in range(ws, we + 1):
                    question_chunks.add(c)

    compress_chunks = set()
    for event in rollout.get("compression_events", []):
        trigger = event.get("trigger_chunk", -1)
        if trigger < 0:
            continue
        for c in range(max(0, trigger - COMPRESS_WINDOW),
                       min(num_chunks, trigger + COMPRESS_WINDOW + 1)):
            compress_chunks.add(c)
        cc = sorted(event.get("compressed_thinks_chunks", []))
        for c in cc[:2] + cc[-2:]:
            compress_chunks.add(c)

    all_selected = all_selected | evidence_chunks | question_chunks | compress_chunks

    # Patrol: fill long gaps
    sorted_sel = sorted(all_selected)
    prev = -1
    for s in sorted_sel:
        if s - prev > 10:
            for c in range(prev + LONG_SILENT_SAMPLE_INTERVAL,
                           s, LONG_SILENT_SAMPLE_INTERVAL):
                chunk_role.setdefault(c, "patrol")
        prev = s
    if sorted_sel and num_chunks - 1 - sorted_sel[-1] > 10:
        for c in range(sorted_sel[-1] + LONG_SILENT_SAMPLE_INTERVAL,
                       num_chunks, LONG_SILENT_SAMPLE_INTERVAL):
            chunk_role.setdefault(c, "patrol")

    # 3. Question windows (medium priority, overrides warmup/patrol)
    for c in question_chunks:
        chunk_role[c] = "question_window"

    # 4. Compress boundaries (medium-high priority)
    for c in compress_chunks:
        chunk_role[c] = "compress_boundary"

    # 2. Evidence anchors (highest priority — overrides everything)
    for c in evidence_chunks:
        chunk_role[c] = "evidence_anchor"

    return chunk_role


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

    chunk_roles = _select_base_chunks(trajectory, rollout, cards_map)
    samples = []

    # Build queries_state interpolation: for any base chunk, use the
    # queries_state from the nearest preceding key_chunk
    sorted_boundaries = sorted(queries_state_at_chunks.keys())

    for chunk_idx, base_role in sorted(chunk_roles.items()):
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
        compress_event = None
        for event in rollout.get("compression_events", []):
            if event.get("trigger_chunk") == chunk_idx:
                compress_event = event
                break

        if compress_event:
            action = "compress"
            # Pass compress event data through snapshot for _make_sample
            compress_snapshot = {"_compress_event": compress_event}
        else:
            action = "silent"
            compress_snapshot = None

        sample = _make_sample(
            chunk_idx=chunk_idx,
            prompt_type="COMPRESS_PROMPT" if compress_event else "SYSTEM_PROMPT",
            action=action,
            think=think,
            queries=qs,
            snapshot=compress_snapshot,
            trajectory_id=traj_id,
            card_id="",
            sequence_type="base",
        )
        # Attach base_role for training loss weighting
        sample["base_role"] = base_role
        samples.append(sample)

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
    from .cache_version import stage_version_ok
    if not stage_version_ok("3c"):
        return None
    path = SAMPLES_3C_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None
