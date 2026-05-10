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
            "skill": "verify which entity actually appeared", "evidence_window": "past",
            "operation": "entity recall", "ovo_task": "ATR/HLD", "ours_unique": False},
    "P1":  {"family_name": "attribute_memory", "category": "Memory & Tracking",
            "skill": "recall an entity color/material/state", "evidence_window": "past",
            "operation": "attribute recall", "ovo_task": "ATR", "ours_unique": False},
    "HLD1": {"family_name": "unanswerable_memory", "category": "Memory & Tracking",
             "skill": "abstain when the requested visual fact is not evidenced",
             "evidence_window": "past", "operation": "abstention",
             "ovo_task": "HLD", "ours_unique": False},
    "CR1": {"family_name": "cause_effect", "category": "Causal & Intent Reasoning",
            "skill": "explain a visible cause-effect relation", "evidence_window": "past",
            "operation": "cause-effect", "ovo_task": "CRR/ASI", "ours_unique": False},
    "CR2": {"family_name": "temporal_order", "category": "Temporal Understanding",
            "skill": "recover the order of observed events", "evidence_window": "past",
            "operation": "event ordering", "ovo_task": "EPM", "ours_unique": False},
    "CR4": {"family_name": "cross_event_reasoning", "category": "Causal & Intent Reasoning",
            "skill": "combine multiple observations across time", "evidence_window": "cross_time",
            "operation": "multi-evidence reasoning", "ovo_task": "CRR/ASI", "ours_unique": False},
    "CR5": {"family_name": "delayed_clue_resolution", "category": "Memory & Tracking",
            "skill": "hold an ambiguous clue until later evidence resolves it",
            "evidence_window": "cross_time", "operation": "delayed resolution",
            "ovo_task": "EPM/CRR", "ours_unique": True},
    "CRR1": {"family_name": "event_happened_status", "category": "Temporal Understanding",
             "skill": "answer whether a described event has happened yet across probes",
             "evidence_window": "streaming", "operation": "event status",
             "ovo_task": "CRR", "ours_unique": False},
    "M1":  {"family_name": "video_summary", "category": "Global Understanding",
            "skill": "summarize the whole video trajectory", "evidence_window": "global",
            "operation": "summary", "ovo_task": "global", "ours_unique": False},
    "E2":  {"family_name": "next_event", "category": "Temporal Understanding",
            "skill": "wait for and identify the next observable event", "evidence_window": "future",
            "operation": "next-event detection", "ovo_task": "EPM", "ours_unique": False},
    "F6":  {"family_name": "future_state", "category": "Temporal Understanding",
            "skill": "predict the next state from current cues", "evidence_window": "current",
            "operation": "future state", "ovo_task": "FPD", "ours_unique": False},
    "F7":  {"family_name": "step_status", "category": "Temporal Understanding",
            "skill": "answer whether a step has happened by now", "evidence_window": "streaming",
            "operation": "status flip", "ovo_task": "SSR", "ours_unique": False},
    "CR3": {"family_name": "intent_now", "category": "Causal & Intent Reasoning",
            "skill": "infer the current actor intent", "evidence_window": "current",
            "operation": "intent inference", "ovo_task": "ASI", "ours_unique": False},
    "CR7": {"family_name": "object_persistence", "category": "Memory & Tracking",
            "skill": "track an object after occlusion or motion", "evidence_window": "current_to_past",
            "operation": "object tracking", "ovo_task": "OJR/STU", "ours_unique": False},
    "R1":  {"family_name": "visible_reasoning", "category": "Current Perception",
            "skill": "reason over the currently visible scene", "evidence_window": "current",
            "operation": "scene reasoning", "ovo_task": "OJR/STU", "ours_unique": False},
    "ACR1": {"family_name": "current_action_recognition", "category": "Current Perception",
             "skill": "recognize the action currently happening", "evidence_window": "current",
             "operation": "action recognition", "ovo_task": "ACR", "ours_unique": False},
    "STU1": {"family_name": "spatial_temporal_understanding", "category": "Current Perception",
             "skill": "identify current spatial relation, count, or direction",
             "evidence_window": "current", "operation": "spatial/count/direction",
             "ovo_task": "STU", "ours_unique": False},
    "OJR1": {"family_name": "object_relation_judgment", "category": "Current Perception",
             "skill": "judge the relation or state of visible objects",
             "evidence_window": "current", "operation": "object relation",
             "ovo_task": "OJR", "ours_unique": False},
    "F5":  {"family_name": "action_count", "category": "Streaming Agent Actions",
            "skill": "emit cumulative counts for repeated actions", "evidence_window": "streaming",
            "operation": "cumulative counting", "ovo_task": "REC", "ours_unique": True},
    "C1":  {"family_name": "text_readout", "category": "Current Perception",
            "skill": "read exact visible text", "evidence_window": "current",
            "operation": "OCR", "ovo_task": "OCR", "ours_unique": False},
    "PN1": {"family_name": "live_narration", "category": "Streaming Agent Actions",
            "skill": "proactively describe sparse state changes", "evidence_window": "streaming",
            "operation": "live narration", "ovo_task": "streaming_agent", "ours_unique": True},
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
    # Family ids define the question/reasoning skill. Availability difficulty
    # (current/direct, memory_direct, recall, future/wait) is assigned later by
    # pass3b placement, so no family should be interpreted as recall-only.
    "N1":  {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Appearance recall: which entity actually appeared in the video",
            **family_taxonomy("N1")},
    "P1":  {"answer_form": "multiple_choice", "profile": "backward",
            "intent": "Attribute memory: color/material/state of an entity",
            **family_taxonomy("P1")},
    "HLD1": {"answer_form": "multiple_choice", "profile": "backward",
             "intent": "Unanswerable visual fact: answer Unable to answer when the fact is not evidenced",
             **family_taxonomy("HLD1")},
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
    "CRR1": {"answer_form": "binary", "profile": "realtime",
             "intent": "Event status over time: has a described event happened yet",
             **family_taxonomy("CRR1")},
    "M1":  {"answer_form": "descriptive", "profile": "backward",
            "intent": "Video summary",
            **family_taxonomy("M1")},
    # forward (silent_then_response)
    "E2":  {"answer_form": "multiple_choice", "profile": "forward",
            "intent": "Next event: wait for the next observable event",
            **family_taxonomy("E2")},
    "F6":  {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Future prediction: answer from current visual cues",
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
    "ACR1": {"answer_form": "multiple_choice", "profile": "realtime",
             "intent": "Current action recognition: identify what is happening now",
             **family_taxonomy("ACR1")},
    "STU1": {"answer_form": "multiple_choice", "profile": "realtime",
             "intent": "Current spatial/count/direction understanding",
             **family_taxonomy("STU1")},
    "OJR1": {"answer_form": "multiple_choice", "profile": "realtime",
             "intent": "Current object relation judgment",
             **family_taxonomy("OJR1")},
    "F5":  {"answer_form": "number", "profile": "realtime",
            "intent": "Action count: repeated action counting (multi_emit, cumulative)",
            **family_taxonomy("F5")},
    "C1":  {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Text readout: exact visible OCR text as a multiple-choice question",
            **family_taxonomy("C1")},
    # multi_emit
    "PN1": {"answer_form": "descriptive", "profile": "realtime",
            "intent": "Live narration (multi_emit, one description per state_change chunk)",
            **family_taxonomy("PN1")},
}

QUESTION_TYPE_BY_FAMILY = {f: ("multi_emit" if f in ("F5", "F7", "CRR1", "PN1") else "single_emit")
                           for f in FAMILY_RULES}


FAMILY_EXTRA_RULES = {
    "HLD1": """
- HLD1 is an explicit negative/unanswerable card.
- Generate diverse negatives, not one repeated template. Prefer a mix of:
  (1) location/where: "Where did I put ...?", "Where was ...?";
  (2) placement/object: "What did I put in/on ...?";
  (3) state yes/no: "Did I leave/close/open ...?";
  (4) count: "How many ...?";
  (5) color/attribute: "What color/material was ...?";
  (6) before-memory: "Where was ... before I picked/used it?"
- The requested target fact MUST be unsupported by ALL provided evidence above,
  not merely absent from the chosen grounding_frames. If any evidence chunk
  states or strongly implies the answer, do not make that card.
- This is not a future/waiting card: the fact should remain unanswerable from
  the provided video evidence, rather than becoming answerable later.
- The correct option text MUST be exactly "Unable to answer"; canonical_answer
  MUST be exactly "Unable to answer"; gold_emits[0].value MUST be the correct
  letter for the Unable option.
- Use four MC options by default; 2, 3, or 5 options are allowed when that is
  natural for the question.
  Put "Unable to answer" in a varied option slot.
- For state yes/no HLD, still include plausible options, e.g. Yes, No, Unable to
  answer, and one plausible concrete state phrase.
- The other options should be scene-plausible concrete answers, but all
  unsupported by the provided evidence. They may look like realistic distractors
  (locations, objects, colors, counts, yes/no states), but must not be correct.
- Do NOT ask about a visible fact from grounding_frames.
- grounding_frames should be representative historical chunks where the
  requested scene/object/action would be checked if it existed. Use chunks that
  show the relevant location, container, person, object group, or activity;
  do not leave grounding_frames empty and do not use future evidence.
- Do NOT include a non-Unable option if that exact option text appears in the
  grounding evidence.
- Avoid leaking the answer inside the question, e.g. do not ask "What color was
  the red umbrella"; ask about a neutral object/fact instead.
- If you cannot construct a safe unanswerable card, output an empty JSON list.""",
    "F7": """
- F7 must be multi-time status.
- gold_emits MUST contain at least one "No" before the change chunk and at least one "Yes" at/after it.
- Values must be monotonic over time: No ... No, then Yes ... Yes; never Yes before No.
- If the timeline permits, include one later Yes probe after the change has
  clearly left the recent visual window, while keeping the total probes to 3-5.
  This creates a meaningful history-check status question without increasing
  the number of F7 cards.
- grounding_frames should include the change chunk that makes the status become Yes.""",
    "CRR1": """
- CRR1 is a multi-probe event-status card.
- Ask whether a concrete event has happened yet. The event description may be
  in the question, but the answer values must be only "No" before the event is
  evidenced and "Yes" at/after the evidence chunk.
- gold_emits MUST contain at least one No before the event and at least one Yes
  after it. Use 3-5 probe chunks, not every chunk in a long range.
- If the timeline permits, include one later Yes probe after the event has
  left the recent visual window. Prefer concrete event-completion facts,
  before/after object states, or clue-resolution events rather than vague
  progress questions.
- Choose events with enough earlier context that the No probes are meaningful.
- grounding_frames should include the event chunk that changes the status.""",
    "F6": """
- F6 is immediate future prediction from current cues, not wait-until-the-future.
- The question should ask what is likely to happen next or what state will
  result, but gold_emits[0].chunk should be the current cue chunk where the
  prediction is answerable from visible intent/context.
- Keep canonical_answer short; avoid answers that require waiting many later
  chunks to verify in the training label.""",
    "C1": """
- C1 is MC OCR, not short_exact.
- The correct option must be exact visible OCR text from the grounding chunk.
- Prefer questions about text that is visually important and easy to confuse:
  labels, signs, titles, product text, screen text, numbers, or short phrases.
- Use five options when there are enough plausible OCR-like distractors.
- Distractors should be other visible text snippets or plausible OCR-like
  snippets with similar length/case/format; avoid one obvious random option.
- If the text is not actually readable from the evidence, do not make a C1
  card; that belongs to HLD1/Unable instead.""",
    "CR1": """
- CR1 should ask for a visible cause or reason grounded in the video, not a
  generic intent guess.
- Prefer cases where the cause and effect are in different nearby or historical
  chunks, and grounding_frames include both sides.
- Distractors should be plausible alternative causes observed elsewhere or
  plausible but unsupported causes, not random actions.""",
    "CR2": """
- CR2 should test event order with 3-4 visually distinct steps.
- Prefer close temporal order, repeated similar actions, or before/after states
  where a model can confuse which event happened first.
- MC distractors should be alternate orders using the same observed events.""",
    "CR4": """
- CR4 should require combining at least two separated observations. Avoid cards
  answerable from one obvious frame.
- Good targets include final object identity inferred from earlier ingredients,
  outcome inferred from setup plus result, or category inferred from multiple
  visual clues.
- grounding_frames should include the minimal separated chunks needed for the
  multi-evidence answer.""",
    "CR5": """
- CR5 should contain an early ambiguous clue and a later resolving observation.
- Prefer clue-resolution questions where the correct answer is not known from
  the early clue alone.
- grounding_frames should include both the clue chunk and the resolving chunk.
- Distractors should match plausible interpretations of the early clue.""",
    "ACR1": """
- ACR1 asks what action is currently visible. Do not ask about intent, cause, or future outcome.""",
    "STU1": """
- STU1 asks a spatial relation, count, direction, or location question grounded
  in a concrete visual moment.
- Prefer event-anchored wording that remains valid if asked later, e.g. "when
  I was ...", "before/after I ...", or "while the ... was visible".
- Do not ask broad scene reasoning.""",
    "OJR1": """
- OJR1 asks the relation/state between visible objects in a concrete moment.
- Prefer event-anchored wording that remains valid if asked later, especially
  object placement, containment, contact, relative position, or state after an
  interaction.
- Do not ask broad scene reasoning.""",
}


# ---------------------------------------------------------------------------
# pass3a — single-family card generation prompt
# ---------------------------------------------------------------------------


def _short_text(value, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    return text[:limit]


def _item_text(item, keys, limit: int) -> str:
    if isinstance(item, dict):
        parts = [
            _short_text(item.get(k, ""), limit)
            for k in keys
            if _short_text(item.get(k, ""), limit)
        ]
        return " ".join(parts)[:limit]
    return _short_text(item, limit)


def _join_items(items, keys, *, max_items: int, item_limit: int) -> str:
    vals = []
    for item in list(items or [])[:max_items]:
        text = _item_text(item, keys, item_limit)
        if text:
            vals.append(text)
    return "; ".join(vals)


def _evidence_timeline_line(cap: Dict) -> str:
    c = cap.get("chunk_idx", 0)
    ents = _join_items(
        cap.get("visible_entities") or [],
        ("id", "desc", "action"),
        max_items=5,
        item_limit=70,
    )
    facts = _join_items(
        cap.get("atomic_facts") or [],
        ("fact", "text"),
        max_items=4,
        item_limit=110,
    )
    ocr = _join_items(
        cap.get("ocr") or [],
        ("text",),
        max_items=4,
        item_limit=50,
    )
    changes = _join_items(
        cap.get("state_changes") or [],
        ("text", "fact", "change"),
        max_items=3,
        item_limit=90,
    )
    spatial = _short_text(cap.get("spatial", ""), 120)
    think = _short_text(cap.get("think", ""), 180)
    fields = []
    if ents:
        fields.append(f"entities=({ents})")
    if facts:
        fields.append(f"facts=({facts})")
    if spatial:
        fields.append(f"spatial=({spatial})")
    if ocr:
        fields.append(f"ocr=({ocr})")
    if changes:
        fields.append(f"changes=({changes})")
    if think:
        fields.append(f"think=({think})")
    line = f"  [c{c}] " + " ".join(fields)
    if len(line.strip()) <= 5:
        line = f"  [c{c}] no salient pass1 facts"
    return line


def _evidence_salience_score(cap: Dict, line: str) -> int:
    """Score chunks for prompt-budget compaction without losing coverage."""
    score = 0
    score += 8 * min(len(cap.get("ocr") or []), 4)
    score += 6 * min(len(cap.get("state_changes") or []), 3)
    score += 4 * min(len(cap.get("atomic_facts") or []), 4)
    score += 2 * min(len(cap.get("visible_entities") or []), 5)
    if cap.get("spatial"):
        score += 3
    if cap.get("think"):
        score += 1
    text = line.lower()
    for marker in (
        "text", "reads", "number", "count", "color", "before", "after",
        "moves", "places", "opens", "closes", "changes", "visible state",
    ):
        if marker in text:
            score += 2
    return score


def _format_evidence_timeline(evidence: List[Dict], *, max_chars: int = 90000) -> str:
    """Compact pass1/pass1b timeline for the card teacher.

    If the full timeline exceeds the prompt budget, keep broad coverage across
    the whole video and fill the remaining budget with high-information chunks.
    This is safer than prefix truncation because backward, future, and
    cross-event cards need late evidence too.
    """
    rows = []
    for pos, cap in enumerate(sorted(evidence, key=lambda x: int(x.get("chunk_idx", 0)))):
        line = _evidence_timeline_line(cap)
        rows.append((
            pos,
            int(cap.get("chunk_idx", pos)),
            line,
            _evidence_salience_score(cap, line),
        ))
    if not rows:
        return ""
    full_text = "\n".join(row[2] for row in rows)
    if len(full_text) <= max_chars:
        return full_text

    note = (
        f"  ... compacted full-video timeline: kept {{kept}}/{len(rows)} chunks "
        f"from c{rows[0][1]}-c{rows[-1][1]}; omitted lower-salience chunks "
        "only because the teacher prompt budget was exceeded ..."
    )
    budget = max(4000, max_chars - len(note) - 16)
    selected: set[int] = set()
    used = 0

    def try_add(idx: int) -> bool:
        nonlocal used
        if idx < 0 or idx >= len(rows) or idx in selected:
            return False
        line_len = len(rows[idx][2]) + 1
        if used + line_len > budget:
            return False
        selected.add(idx)
        used += line_len
        return True

    # Always expose the global endpoints, then add a best chunk from each
    # temporal bucket so late-video evidence is not starved by early chunks.
    try_add(0)
    try_add(len(rows) - 1)
    bucket_count = min(len(rows), max(12, min(72, max_chars // 1500)))
    for b in range(bucket_count):
        lo = int(b * len(rows) / bucket_count)
        hi = int((b + 1) * len(rows) / bucket_count)
        if hi <= lo:
            hi = min(len(rows), lo + 1)
        mid = (lo + hi - 1) / 2.0
        best_idx = max(
            range(lo, hi),
            key=lambda i: (rows[i][3], -abs(i - mid), -rows[i][1]),
        )
        try_add(best_idx)

    # Fill remaining budget with high-salience chunks anywhere in the video.
    for idx in sorted(range(len(rows)), key=lambda i: (-rows[i][3], rows[i][1])):
        if not try_add(idx) and used >= budget:
            break

    kept = sorted(selected)
    return "\n".join([note.format(kept=len(kept))] + [rows[i][2] for i in kept])


def _generation_guidance(rule: Dict, qtype: str) -> str:
    profile = rule.get("profile", "")
    answer_form = rule.get("answer_form", "")
    if qtype == "multi_emit":
        return """
Generation guidance for streaming multi-answer cards:
- Ask one stable question whose answer can be updated at several probe chunks.
- F5/counting: use a repeated action and emit the cumulative count at each occurrence.
- F7/status: choose a concrete step/state that is false before a change and true after it.
- Keep the active span useful but not gratuitously long; do not include dense per-frame emits.
"""
    if profile == "backward":
        return """
Generation guidance for past-memory/cross-time cards:
- You may ask about entities, attributes, places, OCR, object relations, action order,
  before/after state, or a causal clue that is visible earlier in the timeline.
- Use the full timeline above to decide what really existed and to make distractors
  that are plausible but wrong.
- grounding_frames should be the minimal earlier chunks needed for the answer, not
  every chunk where the object remains visible.
"""
    if profile == "forward":
        return """
Generation guidance for future/next-event cards:
- Ask about the next observable event or state that becomes clear later in the timeline.
- The answer should become determinable in a short future window and should be concise:
  one event/state phrase or one MC letter, not a long description.
- Avoid questions whose answer requires keeping the query open across many unrelated
  later events, because long unresolved questions block the next question.
"""
    if answer_form == "multiple_choice":
        return """
Generation guidance for current/direct cards:
- Ask about what is visible at the grounding chunk: action, spatial relation, count,
  object relation, OCR, or current intent if it is visually supported.
- Distractors should be visually plausible alternatives from nearby or other chunks,
  not generic random words.
"""
    return """
Generation guidance:
- Ask a natural visual question whose answer is directly supported by the listed chunks.
- Keep the canonical answer short enough to be used as a streaming response.
"""


def _question_style_guidance(rule: Dict, qtype: str) -> str:
    answer_form = rule.get("answer_form", "")
    parts = [
        "Question and option style guidance:",
        "- Use compact closed-form video-QA wording that a user could ask while watching.",
        "- Keep most questions answerable by one clear visual judgment, but keep some",
        "  exploratory: combine separated clues, compare before/after state, track an",
        "  object through a change, or ask whether evidence is insufficient.",
        "- Vary the surface form. Use direct forms such as what/which/where/how many,",
        "  status forms such as did/has/by now, and anchored history forms such as",
        "  before/after/while/when. Do not repeat one template across cards.",
        "- The user-facing question should be short and specific. Avoid long setup",
        "  paragraphs, hidden answer-format instructions, and repeated phrases like",
        "  'based on the video' unless it is the most natural wording.",
        "- Use natural event anchors instead of internal time: 'after the bowl was",
        "  moved', 'while the label was visible', 'before the object was picked up'.",
    ]
    if answer_form == "multiple_choice":
        parts.extend([
            "- Multiple-choice options should look like a real hard choice, not random",
            "  distractors: same semantic type, similar specificity, similar length when",
            "  possible, and no grammar or length giveaway.",
            "- Prefer distractors from nearby events, visually similar objects/actions,",
            "  plausible before/after states, or similar OCR snippets. For exploratory",
            "  cards, use distractors that correspond to tempting but wrong evidence.",
            "- Use four options by default. Use five options only when the fifth option",
            "  adds a meaningful close distractor or an Unable-to-answer alternative.",
        ])
    elif answer_form == "binary":
        parts.extend([
            "- Binary questions should make the decision boundary explicit: the event",
            "  has either happened by the probe moment or has not happened yet.",
            "- Avoid vague progress wording; name the concrete event or state transition.",
        ])
    elif answer_form == "number":
        parts.extend([
            "- Number questions should specify the counted unit clearly and use evidence",
            "  chunks where cumulative updates are visually distinguishable.",
        ])
    if qtype == "multi_emit":
        parts.extend([
            "- For multi-answer cards, the same question must remain stable across all",
            "  probes; only the answer changes as the stream progresses.",
        ])
    return "\n".join(parts)


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

    evidence_text = _format_evidence_timeline(evidence)
    generation_guidance = _generation_guidance(rule, qtype).strip()
    style_guidance = _question_style_guidance(rule, qtype).strip()

    options_block = ""
    if rule["answer_form"] == "multiple_choice":
        options_block = """
  "options": ["A) ...", "B) ...", "..."],                # 2-5 plausible options; 4 by default, 5 allowed with "E) ..."
  "correct_option": "A" | "B" | "C" | "D" | "E",          # the gold letter"""

    if qtype == "multi_emit":
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}, ...],   '
                     '# multi_emit: ONE entry per occurrence/event chunk; '
                     '"value" is cumulative count for F5, Yes/No status for F7, '
                     'or "event@N" for PN1')
    else:
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}],         '
                     '# single_emit: exactly 1 entry; chunk = when answer is determinable')
    family_extra = FAMILY_EXTRA_RULES.get(family, "").strip()

    return f"""You are a teacher generating training card(s) from a video's per-chunk evidence.

Family: {family}  ({rule["intent"]})
Category: {rule["category"]} / {rule["family_name"]}
Answer form: {rule["answer_form"]}
Question type: {qtype}

Evidence timeline from pass1/pass1b (per-chunk visible_entities + facts + spatial + ocr + state_changes + think):
{evidence_text}

{generation_guidance}

{style_guidance}

Produce {target_n} card(s) as a JSON list. Each card schema:
{{
  "family": "{family}",
  "question": "...",                                     # bare natural-language question only
  "answer_form": "{rule['answer_form']}",
  "canonical_answer": "...",                              # the final/correct answer text{options_block}
  {emits_doc}
  "grounding_frames": [int, ...]                         # MINIMAL set of chunk indices needed to verify the answer
}}

Rules:
- question must NOT contain or paraphrase the answer.
- question must contain ONLY the user question. Do NOT put options, answer
  format instructions, choice prompts, response-format phrases, or letter
  labels inside question. Options and answer requirements belong only in the
  structured fields below so later rendering can decide how to show them.
- Do not mention benchmark names, family ids, or dataset/task labels in the
  user-facing question.
- question must be a natural user-facing question. Do NOT mention internal
  chunk indices, frame numbers, timestamps, "c12", "chunk 12", or evidence
  row ids. Use visual/event references instead.
- grounding_frames must reference chunks present in the evidence above.
- Treat the family as a reasoning type, not an availability bucket. The same
  card may later be placed as current/direct, memory_direct, or recall.
- If producing multiple cards for this family, make them semantically diverse:
  use different events/chunks and different answer types within the family.
  Prefer one immediately visible/current-style card and one event-anchored
  historical-detail card when the evidence supports both. Do not make near
  duplicates with only different options.
- Prefer questions whose evidence remains meaningful under harder placement:
  fine visual details, before/after state, event order, causal clue, OCR,
  object relation, or multi-chunk support when the family permits it.
- For historical/recall-capable cards, prefer facts that are likely to be lost
  or blurred by text memory/compression: exact object relation, small visual
  attribute, OCR text, before/after object state, action order, count, or
  relation between two objects. Avoid overly coarse questions whose answer is
  just the main object/action already stated in the per-chunk think.
- If a question may be asked after the evidence is no longer visible, anchor it
  to a natural past event ("when I ...", "before/after ...", "while ...") so
  the gold answer time remains determined by the grounding evidence, not by the
  later recall/action decision.
- For MC: distractors must be PLAUSIBLE, not random. For non-HLD families,
  prefer distractors drawn from other observed entities/actions in the video.
  For HLD1, distractors should be scene-plausible but unsupported.
- MC options should be mutually exclusive and the same semantic type as the
  correct answer: all locations, all objects, all actions, all counts, or all
  state phrases. Avoid synonyms of the correct answer, "all/none of the above",
  joke options, length giveaways, or one option that is much more specific than
  the others. Exactly one option should be correct.
- For binary: canonical_answer ∈ {{"Yes", "No"}}.
- For number: canonical_answer is a digit string.
- For short_exact: canonical_answer is ≤ 4 words.
- For descriptive: canonical_answer is 1-3 sentences grounded in evidence.
{family_extra}

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


def recall_query_prompt(
    card: Dict,
    *,
    current_chunk: int | None = None,
    mode: str = "answer",
    reason: str = "",
) -> str:
    """Generate retrieval keywords for a historical visual-recall card.

    ``mode='answer'`` is the normal recall+response path. Waiting/silent
    recall probes are generated by a deterministic query builder to avoid
    calling the teacher at every pending timestep.
    """
    grounding = card.get("grounding_frames") or []
    if current_chunk is not None:
        grounding = [g for g in grounding if int(g) < int(current_chunk)]
    if grounding:
        from ..config import AGENT_CHUNK_SEC
        tr = f"{int(min(grounding) * AGENT_CHUNK_SEC)}-{int((max(grounding) + 1) * AGENT_CHUNK_SEC)}"
    else:
        tr = ""
    current_doc = (
        f"\nCurrent ask chunk: c{int(current_chunk)}" if current_chunk is not None else ""
    )
    reason_doc = f"\nRecall reason: {reason}" if reason else ""
    hld_doc = ""
    if card.get("family") == "HLD1":
        hld_doc = (
            "\nHLD/Unable case: the query should check historical evidence for "
            "absence/unsupported status. Include the requested target object or "
            "state plus broad scene anchors such as visible objects/location; "
            "do not include 'Unable to answer' or any option letter."
        )
    return f"""Generate a retrieval query for this historical-recall question.

Question: {card.get('question', '')}
Approximate time range of evidence: {tr or 'unknown'}
Mode: {mode}{current_doc}{reason_doc}{hld_doc}

The query is used only to FIND the past evidence. It must not contain the
answer itself.

Rules:
- Output 3-5 discriminative keywords: visible entity descriptions, scene
  anchors, object names, and actions near the evidence.
- Do NOT include answer values, correct-option text, exact OCR/number/color
  values being asked for, or words that trivially reveal the answer.
- For cumulative or "so far" questions, search for the repeated action or
  object being counted, not for the final count.
- For status/history questions, search for the event or object whose earlier
  occurrence determines the answer, not for "yes" or "no".
- If the question asks "what text/number/color/state/count", query for the
  surrounding object/action/location instead of the target value.
- NO pronouns, NO articles, NO full sentence.
- time_range must be the provided historical range and must end before the
  current ask chunk.

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
