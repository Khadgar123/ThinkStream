"""397B prompts for v2 pass3a card generation + pass3c response/recall.

Schema produced by pass3a_card_prompt MUST match v2.design.Card so
v2.cards._dict_to_card / pass3a._card_to_dict round-trip cleanly.

All prompts are pure-string templates with .format(...) substitution; the
caller (pass3a_cards.py / pass3c_samples.py) wraps them into a vLLM request.
"""

from __future__ import annotations

import json
from typing import Dict, List


# ---------------------------------------------------------------------------
# Family taxonomy + per-family generation rules (mirrors v2/cards.py)
# ---------------------------------------------------------------------------

# Keep the compact legacy ids (N1/F5/PN1/...) as stable primary keys because
# placement, cached cards, rewards, and eval rows already reference them.
# The new fields below are the human-facing taxonomy used for data audits,
# paper tables, and downstream sampling. They are inspired by StreamO/OvO
# style buckets, but include ThinkStream-specific streaming-agent skills:
# live narration, tool recall, compression, and silence timing.
FAMILY_TAXONOMY = {
    "N1":  {"family_name": "appearance_recall", "category": "Memory & Tracking",
            "skill": "verify which entity actually appeared", "ours_unique": False},
    "P1":  {"family_name": "attribute_memory", "category": "Memory & Tracking",
            "skill": "recall an entity color/material/state", "ours_unique": False},
    "CR1": {"family_name": "cause_effect", "category": "Causal & Intent Reasoning",
            "skill": "explain a visible cause-effect relation", "ours_unique": False},
    "CR2": {"family_name": "temporal_order", "category": "Temporal Understanding",
            "skill": "recover the order of observed events", "ours_unique": False},
    "CR4": {"family_name": "cross_event_reasoning", "category": "Causal & Intent Reasoning",
            "skill": "combine multiple observations across time", "ours_unique": False},
    "CR5": {"family_name": "delayed_clue_resolution", "category": "Memory & Tracking",
            "skill": "hold an ambiguous clue until later evidence resolves it", "ours_unique": True},
    "M1":  {"family_name": "video_summary", "category": "Global Understanding",
            "skill": "summarize the whole video trajectory", "ours_unique": False},
    "E2":  {"family_name": "next_event", "category": "Temporal Understanding",
            "skill": "wait for and identify the next observable event", "ours_unique": False},
    "F6":  {"family_name": "future_state", "category": "Temporal Understanding",
            "skill": "predict the next state from current evidence", "ours_unique": False},
    "F7":  {"family_name": "step_status", "category": "Progress Monitoring",
            "skill": "answer whether a step has happened by now", "ours_unique": False},
    "CR3": {"family_name": "intent_now", "category": "Causal & Intent Reasoning",
            "skill": "infer the current actor intent", "ours_unique": False},
    "CR7": {"family_name": "object_persistence", "category": "Memory & Tracking",
            "skill": "track an object after occlusion or motion", "ours_unique": False},
    "R1":  {"family_name": "visible_reasoning", "category": "Current Perception",
            "skill": "reason over the currently visible scene", "ours_unique": False},
    "F5":  {"family_name": "action_count", "category": "Streaming Agent Actions",
            "skill": "emit cumulative counts for repeated actions", "ours_unique": True},
    "C1":  {"family_name": "text_readout", "category": "Current Perception",
            "skill": "read exact visible text", "ours_unique": False},
    "PN1": {"family_name": "live_narration", "category": "Streaming Agent Actions",
            "skill": "proactively describe sparse state changes", "ours_unique": True},
}


def family_taxonomy(family: str) -> Dict:
    """Return human-facing taxonomy fields for a stable family id."""
    return dict(FAMILY_TAXONOMY.get(family, {
        "family_name": family or "unknown",
        "category": "Unknown",
        "skill": "",
        "ours_unique": False,
    }))


FAMILY_RULES = {
    # backward MC (recall_demo dominant)
    "N1":  {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Appearance recall: which entity actually appeared in the video",
            **family_taxonomy("N1")},
    "P1":  {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Attribute memory: color/material/state of an entity",
            **family_taxonomy("P1")},
    "CR1": {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Cause-effect: why did X happen given visible cause",
            **family_taxonomy("CR1")},
    "CR2": {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Temporal ordering: what was the sequence of N events",
            **family_taxonomy("CR2")},
    "CR4": {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Cross-event reasoning: combine 2+ observations to derive answer",
            **family_taxonomy("CR4")},
    "CR5": {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Delayed clue resolution: an early clue is resolved by later evidence",
            **family_taxonomy("CR5")},
    "M1":  {"answer_form": "descriptive", "profile": "backward",
            "intent": "Video summary",
            **family_taxonomy("M1")},
    # forward (silent_then_response)
    "E2":  {"answer_form": "multiple_choice", "profile": "forward",
            "intent": "Next event: wait for the next observable event",
            **family_taxonomy("E2")},
    "F6":  {"answer_form": "multiple_choice", "profile": "forward",
            "intent": "Future state: predict next state given current",
            **family_taxonomy("F6")},
    "F7":  {"answer_form": "binary", "profile": "realtime",
            "intent": "Step status: has step X happened yet (Yes/No flips at step_chunk)",
            **family_taxonomy("F7")},
    # realtime (direct)
    "CR3": {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Intent now: what is the actor trying to do now",
            **family_taxonomy("CR3")},
    "CR7": {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Object persistence: where is the currently occluded/tracked object",
            **family_taxonomy("CR7")},
    "R1":  {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Visible reasoning: reason about the current scene",
            **family_taxonomy("R1")},
    "F5":  {"answer_form": "number", "profile": "realtime",
            "intent": "Action count: repeated action counting (multi_emit, cumulative)",
            **family_taxonomy("F5")},
    "C1":  {"answer_form": "short_exact", "profile": "realtime",
            "intent": "Text readout: exact text visible on screen",
            **family_taxonomy("C1")},
    # multi_emit
    "PN1": {"answer_form": "descriptive", "profile": "realtime",
            "intent": "Live narration (multi_emit, one description per state_change chunk)",
            **family_taxonomy("PN1")},
}

QUESTION_TYPE_BY_FAMILY = {f: ("multi_emit" if f in ("F5", "PN1") else "single_emit")
                           for f in FAMILY_RULES}


# ---------------------------------------------------------------------------
# pass3a — single-family card generation prompt
# ---------------------------------------------------------------------------


def card_generation_prompt(
    family: str,
    evidence: List[Dict],
    target_n: int = 1,
) -> str:
    """Build a 397B prompt that produces v2 cards for a single family.

    Returns a JSON-instructed prompt. The teacher is asked to emit a JSON
    list whose entries match the v2 schema.
    """
    rule = FAMILY_RULES[family]
    qtype = QUESTION_TYPE_BY_FAMILY[family]

    # Compact evidence representation: chunk → (entities + facts + ocr + state_changes)
    ev_lines = []
    for cap in evidence[:80]:                  # cap to keep prompt bounded
        c = cap.get("chunk_idx", 0)
        ents = "; ".join(
            (e.get("desc", "")[:40] + ("[" + e.get("action", "")[:20] + "]"
                                       if e.get("action") else ""))
            for e in (cap.get("visible_entities") or [])[:3]
        )
        facts = "; ".join(
            f.get("fact", "")[:80] for f in (cap.get("atomic_facts") or [])[:2]
        )
        ocr = "/".join(cap.get("ocr") or [])[:40]
        sc = "/".join(
            (s if isinstance(s, str) else s.get("text", str(s)))[:60]
            for s in (cap.get("state_changes") or [])
        )
        ev_lines.append(
            f"  [c{c}] entities=({ents}) facts=({facts})"
            + (f" ocr=({ocr})" if ocr else "")
            + (f" change=({sc})" if sc else "")
        )
    evidence_text = "\n".join(ev_lines)

    options_block = ""
    if rule["answer_form"] == "multiple_choice":
        options_block = """
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],   # 4 plausible options, EXACTLY one correct
  "correct_option": "A" | "B" | "C" | "D",                # the gold letter"""

    if qtype == "multi_emit":
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}, ...],   '
                     '# multi_emit: ONE entry per occurrence/event chunk; '
                     '"value" is cumulative count for F5 or "event@N" for PN1')
    else:
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}],         '
                     '# single_emit: exactly 1 entry; chunk = when answer is determinable')

    return f"""You are a teacher generating ONE training card from a video's per-chunk evidence.

Family: {family}  ({rule["intent"]})
Category: {rule["category"]} / {rule["family_name"]}
Answer form: {rule["answer_form"]}
Question type: {qtype}

Evidence (per-chunk visible_entities + atomic_facts + ocr + state_changes):
{evidence_text}

Produce {target_n} card(s) as a JSON list. Each card schema:
{{
  "family": "{family}",
  "question": "...",                                     # natural-language question
  "answer_form": "{rule['answer_form']}",
  "canonical_answer": "...",                              # the final/correct answer text{options_block}
  {emits_doc}
  "grounding_frames": [int, ...]                         # MINIMAL set of chunk indices needed to verify the answer
}}

Rules:
- question must NOT contain or paraphrase the answer.
- grounding_frames must reference chunks present in the evidence above.
- For MC: distractors must be PLAUSIBLE (drawn from other observed entities/actions in the video), not random.
- For binary: canonical_answer ∈ {{"Yes", "No"}}.
- For number: canonical_answer is a digit string.
- For short_exact: canonical_answer is ≤ 4 words.
- For descriptive: canonical_answer is 1-3 sentences grounded in evidence.

Output ONLY a JSON list, no commentary:"""


# ---------------------------------------------------------------------------
# pass3c — response generation prompt
# ---------------------------------------------------------------------------


def response_generation_prompt(card: Dict, ask_chunk: int) -> str:
    """Ask 397B to write the assistant's response text.

    For MC/binary/number/short_exact the answer is emitted deterministically
    by pass3c. LLM rewriting is only used for descriptive single-emit cards.
    """
    af = card.get("answer_form", "")
    if af in ("multiple_choice", "binary", "number", "short_exact"):
        # Direct emission — no LLM call needed; caller short-circuits
        return ""
    return f"""Write a concise streaming-agent response to this question.

Question: {card.get('question', '')}
Gold answer (must convey this): {card.get('canonical_answer', '')}
Answer form: {af}

Rules:
- Length: 1-2 sentences.
- Match the gold answer faithfully; do not introduce extra facts.
- Use streaming-agent voice (concise, factual, present tense).

Output the response text ONLY, no quotes or prefix:"""


# ---------------------------------------------------------------------------
# pass3c — recall_query generation prompt
# ---------------------------------------------------------------------------


def recall_query_prompt(card: Dict) -> str:
    """Generate retrieval keywords for recall demo (backward profile cards)."""
    grounding = card.get("grounding_frames") or []
    if grounding:
        from ..config import AGENT_CHUNK_SEC
        tr = f"{int(min(grounding) * AGENT_CHUNK_SEC)}-{int((max(grounding) + 1) * AGENT_CHUNK_SEC)}"
    else:
        tr = ""
    return f"""Generate a retrieval query for this question.

Question: {card.get('question', '')}
Approximate time range of evidence: {tr or 'unknown'}

Output 3-5 discriminative keywords (entity descriptions + action anchors).
NO answer values, NO pronouns, NO articles.

Output JSON ONLY (one line):
{{"query": "keyword1 keyword2 keyword3", "time_range": "{tr}"}}"""


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def parse_card_response(raw: str, family: str) -> List[Dict]:
    """Best-effort parse a 397B response into a list of v2 card dicts.

    Strips markdown fences, finds the first JSON list, validates required
    keys. Returns empty list on parse failure (caller should log + fallback).
    """
    if not raw:
        return []
    text = raw.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[-2] if text.count("```") >= 2 else text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    # Find the first '[' ... matching ']'
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return []
    try:
        cards = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(cards, list):
        return []
    valid = []
    required = {"family", "question", "answer_form", "gold_emits", "grounding_frames"}
    for c in cards:
        if not isinstance(c, dict):
            continue
        if not required.issubset(c.keys()):
            continue
        c["family"] = family    # enforce
        # Normalize gold_emits
        emits = c.get("gold_emits") or []
        if not isinstance(emits, list):
            continue
        norm_emits = []
        for e in emits:
            if isinstance(e, dict) and "chunk" in e and "value" in e:
                norm_emits.append({"chunk": int(e["chunk"]),
                                    "value": str(e["value"])})
        if not norm_emits:
            continue
        c["gold_emits"] = norm_emits
        c["grounding_frames"] = [int(g) for g in (c.get("grounding_frames") or [])]
        c.setdefault("canonical_answer",
                     norm_emits[-1]["value"] if norm_emits else "")
        c["question_type"] = QUESTION_TYPE_BY_FAMILY.get(family, "single_emit")
        c.update(family_taxonomy(family))
        valid.append(c)
    return valid


def parse_recall_query_response(raw: str, fallback_time_range: str = "") -> Dict:
    if not raw:
        return {"query": "", "time_range": fallback_time_range}
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        return {"query": "", "time_range": fallback_time_range}
    try:
        d = json.loads(text[start:end + 1])
        return {
            "query": str(d.get("query", "")),
            "time_range": str(d.get("time_range", fallback_time_range)),
        }
    except (json.JSONDecodeError, ValueError):
        return {"query": "", "time_range": fallback_time_range}
