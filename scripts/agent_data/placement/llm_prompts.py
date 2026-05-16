"""397B prompts for v2 pass3a card generation + pass3c response/recall.

Schema produced by pass3a_card_prompt MUST match v2.design.Card so
v2.cards._dict_to_card / pass3a._card_to_dict round-trip cleanly.

All prompts are pure-string templates with .format(...) substitution; the
caller (pass3a_cards.py / pass3c_samples.py) wraps them into a vLLM request.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


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
             "skill": "answer whether current evidence can resolve a previous question",
             "evidence_window": "streaming", "operation": "evidence sufficiency status",
             "ovo_task": "CRR", "ours_unique": False},
    "M1":  {"family_name": "video_summary", "category": "Global Understanding",
            "skill": "summarize the whole video trajectory", "evidence_window": "global",
            "operation": "summary", "ovo_task": "global", "ours_unique": True},
    "E2":  {"family_name": "proactive_output", "category": "Temporal Understanding",
            "skill": "wait for a future visual trigger and output a short target",
            "evidence_window": "future",
            "operation": "proactive trigger output", "ovo_task": "EPM", "ours_unique": False},
    "F6":  {"family_name": "future_state", "category": "Temporal Understanding",
            "skill": "predict the next state from current cues", "evidence_window": "current",
            "operation": "future state", "ovo_task": "FPD", "ours_unique": False},
    "F7":  {"family_name": "step_status", "category": "Temporal Understanding",
            "skill": "answer whether a step is currently being carried out",
            "evidence_window": "streaming", "operation": "current step status",
            "ovo_task": "SSR", "ours_unique": False},
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
            "operation": "cumulative counting", "ovo_task": "REC", "ours_unique": False},
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
    # (current/direct, state-memory direct, recall, future/wait) is assigned
    # later by pass3b placement, so no family should be interpreted as
    # recall-only.
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
             "intent": "CRR probe status: whether current evidence now resolves a prior visual question",
             **family_taxonomy("CRR1")},
    "M1":  {"answer_form": "descriptive", "profile": "backward",
            "intent": "Video summary",
            **family_taxonomy("M1")},
    # forward (silent_then_response)
    "E2":  {"answer_form": "short_exact", "profile": "forward",
            "intent": "Proactive output: wait for a future trigger and emit the requested short output",
            **family_taxonomy("E2")},
    "F6":  {"answer_form": "multiple_choice", "profile": "realtime",
            "intent": "Future prediction: answer from current visual cues",
            **family_taxonomy("F6")},
    "F7":  {"answer_form": "binary", "profile": "realtime",
            "intent": "SSR step status: immediate Yes/No check for whether a step is currently being carried out",
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

QUESTION_TYPE_BY_FAMILY = {f: ("multi_emit" if f in ("F5", "CRR1", "PN1") else "single_emit")
                           for f in FAMILY_RULES}


ANSWER_FORM_VARIANTS = {
    # Keep perception/backward tracing families in their benchmark-like MCQ
    # form. Non-MCQ pressure should mainly come from active responding:
    # F5/REC counting, F7/SSR status, and CRR1/CRR status-over-time.
    "E2": ("short_exact", "number"),
}


FAMILY_EXTRA_RULES = {
    "P1": """
- P1 asks for a visible object attribute such as color, material, state, or
  appearance. Do not turn P1 into OCR/text/brand reading; those belong to C1.
- Keep question_way="object_attribute" and evidence_type="object_attribute_visual".
- The correct answer must be directly visible in the planned support chunks.""",
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
- Avoid global absence claims unless the provided evidence truly covers the
  relevant scene. Prefer bounded historical checks such as a location,
  container, object group, or action interval that the grounding frames can
  inspect.
- Avoid leaking the answer inside the question, e.g. do not ask "What color was
  the red umbrella"; ask about a neutral object/fact instead.
- If you cannot construct a safe unanswerable card, output an empty JSON list.""",
    "F7": """
- F7 mirrors OVO SSR: an immediate current step-status row, not a persistent
  multi-answer state probe and not historical event status.
- The question should ask whether the person is currently carrying out a step,
  e.g. "Is the person currently ...?" Do not use "has happened", "yet", or
  "by now".
- question_type must be single_emit. Use exactly one gold_emit at the current
  probe chunk where the answer is judged.
- answer_form must be binary, with no options. The gold value is "Yes" only
  if the step is currently visible at that probe; otherwise "No".
- grounding_frames should include the same current probe chunk plus any
  immediately adjacent current evidence needed to verify the status.""",
    "CRR1": """
- CRR1 mirrors OVO CRR: a probe asks whether the video-so-far/current prefix
  now provides enough information to answer an original visual question.
- The user-facing question should be a compact sufficiency/status check such
  as "Can you answer what happened to the cup now?" or "Is there enough visual
  evidence now to answer where the item ended up?"
- The question should not ask for the final hidden answer directly. It asks
  whether the current/latest visible evidence is sufficient yet. Natural
  wording should signal that the answer can change from No to Yes as more
  video arrives.
- Choose an underlying question/clue whose answer is NOT visually resolvable
  at the earliest planned probe but becomes resolvable at a later planned
  probe. Do not choose a fact that is already visible at the first probe; that
  would make all probe values "Yes" and is not a valid CRR1 card.
- Answer "No" before the clue/resolution is visually available. Once a planned
  probe is "Yes", every later planned probe must also be "Yes".
- Use 3-5 probe chunks, not every chunk in a long range. One later Yes probe is
  useful, but this is still a status/state task; do not design it as a default
  recall example.
- grounding_frames should include the clue/resolution chunks that make the
  answer possible.""",
    "CR3": """
- CR3 is normally current intent/cause from visible evidence.
- If task_subtype is emotion_context_current, ask a compact StreamingBench-like
  emotion or mood question grounded in visible expression/body language and
  immediate context. Keep it multiple-choice and current, not historical.""",
    "R1": """
- R1 is current visible reasoning.
- If task_subtype is scene_understanding_current, ask a current scene/clip
  understanding MCQ. The answer should be visible from the planned support
  chunks, not a broad whole-video summary.
- If task_subtype is multimodal_alignment, ask whether visible cues, OCR/text,
  object state, action context, or scene description match or contradict each
  other. Keep it grounded in the planned visual chunks; do not require audio
  if the evidence does not contain audio facts.""",
    "N1": """
- N1 asks which entity/person appeared or interacted in the video.
- If task_subtype is person_identity_interaction, ask a person/entity
  interaction question similar to StreamingBench sequential QA, but do not
  require an actual previous question unless the wording is self-contained.
- If task_subtype is sequential_reference, use follow-up-style wording with a
  stable earlier referent, e.g. "the person/object just referred to" or
  "that same item", but include enough visual anchor text that the card can be
  rendered independently in pass3C.""",
    "M1": """
- M1 is a global/scene-summary style question, not OCR, brand recall, or a
  single-object attribute question.
- Keep question_way="scene_summary" and evidence_type="global_context_memory".
- Ask about the overall activity, repeated pattern, or broad trajectory shown
  across the planned support chunks. The answer should be one short, concrete
  sentence rather than a paragraph.
- gold_emits must use the planned answer chunk, which is the latest planned
  support chunk.""",
    "F5": """
- F5 should be phrased as cumulative repeated-action counting:
  "How many times has ... happened by now?" or "How many ... have appeared so far?"
- The counted unit must be visually repeatable and unambiguous. Do not count
  vague activity, camera cuts, or inferred intent.
- gold_emits values must be digit strings and should update only at meaningful
  occurrence chunks, not at every frame.
- Do not skip earlier count-changing occurrences before the first gold_emit;
  the first emitted count should be the first clear occurrence in the selected
  sequence, so a chunk-0 OVO-style query has a coherent cumulative trajectory.
- Keep cumulative counts in the OVO REC range 0-10. If the event would require
  larger numbers, choose a narrower repeated action or a shorter probe sequence.
- When evidence permits, include 6-9 cumulative response points so the rendered
  trajectory matches OVO REC-style repeated scoring. Use fewer only when there
  are not enough clear repeated occurrences in the provided video evidence.
- If task_subtype is global_prefix_count, use OVO REC from-start semantics:
  one question can be asked at the beginning of the trajectory and the answer
  is a cumulative digit at each planned answer chunk. Do not make a local-only
  "right now" count for that slot.
- If task_subtype is local_repeated_count, use a compact local repeated-action
  episode with cumulative counts over the selected occurrence chunks.
- This is a memory/state task, not a visual recall task by default.""",
    "E2": """
- E2 mirrors StreamingBench Proactive Output: ask now, stay silent until a
  future visual trigger appears, then output the requested short phrase/number
  exactly once.
- The question may naturally contain the required output phrase, e.g.
  "When the scoreboard shows AD, output 'Break point'." This is intentional:
  the task is trigger detection, not hidden-answer QA.
- The question must read as a proactive trigger instruction, not as a normal
  visual QA question. Use varied future-trigger phrasing such as a later
  appearance, state change, completion, or condition becoming visible. Do not
  reuse one fixed "When X, output Y" wording for every card.
- Use short_exact or number answers only. Do not use multiple-choice options.
- The trigger must be visually checkable in one future chunk or a short future
  window. Avoid vague long-horizon goals.
- grounding_frames/gold_emits[0].chunk must be the trigger chunk where the
  output should be emitted. support_policy must be future_current_cue and
  recall_eligible must be false.""",
    "F6": """
- F6 is immediate future prediction from current cues, not wait-until-the-future.
- The question should ask what is likely to happen next or what state will
  result, but gold_emits[0].chunk should be the current cue chunk where the
  prediction is answerable from visible intent/context.
- The question must be prospective. It should not ask for a current state,
  current OCR/text, visible object attribute, or current location unless that
  visible cue is explicitly used to ask what will happen next or what result is
  about to follow. If the natural question is only a current fact, omit the F6
  card and leave it for R1/CR3/C1/STU/OJR.
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
- MC distractors should be alternate orders using the same observed events.
- Use only the planned support chunks to establish the before/after relation,
  and emit the answer exactly at the planned answer chunk.
- Do not add new unplanned future chunks to grounding_frames or gold_emits.""",
    "CR4": """
- CR4 should require combining at least two separated observations. Avoid cards
  answerable from one obvious frame.
- Good targets include final object identity inferred from earlier ingredients,
  outcome inferred from setup plus result, or category inferred from multiple
  visual clues.
- grounding_frames should include the minimal separated chunks needed for the
  multi-evidence answer.
- If task_subtype is contextual_misleading_or_anomaly, mirror
  StreamingBench contextual/anomaly style: ask what is actually happening or
  what context is misleading/abnormal, using concrete visual evidence rather
  than generic commonsense.
- If task_subtype is source_discrimination, ask which visible cue/source in
  the planned chunks supports the answer, or distinguish what was visually
  observed from what would only be inferred from context.
- Do not move the answer to a later unplanned chunk, and do not turn this into
  a single-frame OCR/text-reading question.""",
    "CR5": """
- CR5 should contain an early ambiguous clue and a later resolving observation.
- Prefer clue-resolution questions where the correct answer is not known from
  the early clue alone.
- The question should make the clue/resolution relationship natural to the
  viewer: an early ambiguous object/action, a later reveal, or a later outcome
  that settles which interpretation was correct.
- If the planned slot is selected as a future/wait style card, phrase the
  question so it is clear the assistant must wait for the resolving visual
  evidence instead of guessing from the clue. If the planned slot is historical
  recall, phrase it as an earlier completed clue/reveal moment.
- For ambiguous past_or_future_clue slots, choose one temporal stance in the
  question text rather than leaving it ambiguous. A future/wait CR5 question
  must explicitly mention waiting for a later reveal/resolving evidence; a
  historical CR5 question must use before/earlier/completed-event wording. Do
  not write a historical "Before..." question if the intended use is future
  delayed placement.
- When multiple CR5 cards are requested and evidence supports both styles,
  include both a historical clue-recall wording and a future wait-for-reveal
  wording instead of making all CR5 cards historical.
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


def _format_planned_slot_evidence(
    evidence: List[Dict],
    planned_slots: List[Dict],
    *,
    max_lines_per_slot: int = 24,
    max_total_chars: int = 42000,
) -> str:
    """Render slot-local evidence so the teacher does not drift to other chunks."""
    by_chunk = {
        int(cap.get("chunk_idx", pos)): cap
        for pos, cap in enumerate(evidence)
        if isinstance(cap, dict)
    }
    slot_count = max(1, len(planned_slots or []))
    per_slot_line_budget = max(8, min(max_lines_per_slot, 48 // slot_count))
    remaining_chars = max_total_chars
    blocks: List[str] = []
    for slot in planned_slots:
        support = sorted({int(c) for c in (slot.get("support_chunks") or [])})
        answers = sorted({int(c) for c in (slot.get("answer_chunks") or [])})
        allowed = sorted(set(support) | set(answers))
        shown = allowed
        if len(shown) > per_slot_line_budget:
            # Keep answer chunks, endpoints, and evenly spaced support chunks.
            keep = set(answers)
            if allowed:
                keep.update({allowed[0], allowed[-1]})
            remaining_budget = max(0, per_slot_line_budget - len(keep))
            if remaining_budget:
                last = len(allowed) - 1
                for i in range(remaining_budget):
                    keep.add(allowed[round(i * last / max(1, remaining_budget - 1))])
            shown = [c for c in allowed if c in keep]
        lines = [
            f"- slot_id={slot.get('slot_id', '')}",
            f"  allowed_chunks={allowed}",
            f"  answer_chunks={answers}",
            "  allowed_evidence:",
        ]
        for chunk in shown:
            cap = by_chunk.get(chunk)
            if cap is not None:
                lines.append(_evidence_timeline_line(cap))
        omitted = len(allowed) - len(shown)
        if omitted > 0:
            lines.append(f"  ... omitted {omitted} lower-priority allowed chunks ...")
        block = "\n".join(lines)
        if len(block) > remaining_chars:
            blocks.append(
                "\n".join([
                    f"- slot_id={slot.get('slot_id', '')}",
                    f"  allowed_chunks={allowed}",
                    f"  answer_chunks={answers}",
                    "  ... omitted slot-local evidence because the prompt budget was reached ...",
                ])
            )
            break
        blocks.append(block)
        remaining_chars -= len(block) + 2
    return "\n\n".join(blocks)


def _format_planned_slot_guidance(planned_slots: List[Dict]) -> str:
    lines: List[str] = []
    for slot in planned_slots:
        slot_id = str(slot.get("slot_id", ""))
        task_family = str(slot.get("task_family") or slot.get("slot_group") or "")
        task_subtype = str(slot.get("task_subtype") or slot.get("slot_subtype") or "")
        timing_type = str(slot.get("timing_type") or slot.get("temporal_bucket") or "")
        readable = str(slot.get("readable_task_name") or " / ".join(
            p for p in (task_family, task_subtype, timing_type) if p
        ))
        legacy = str(slot.get("legacy_family_id") or slot.get("family") or "")
        goal = str(slot.get("question_goal", ""))
        behavior = str(slot.get("answer_behavior", ""))
        hint = str(slot.get("placement_hint", ""))
        if not any((readable, goal, behavior, hint)):
            continue
        lines.append(
            f"- {slot_id}: task={readable}; legacy_family_id={legacy}; "
            f"answer_behavior={behavior}; goal={goal}; placement_hint={hint}"
        )
    return "\n".join(lines)


def _generation_guidance(rule: Dict, qtype: str) -> str:
    profile = rule.get("profile", "")
    answer_form = rule.get("answer_form", "")
    if qtype == "multi_emit":
        return """
Generation guidance for streaming multi-answer cards:
- Ask one stable question whose answer can be updated at several probe chunks.
- F5/counting: use a repeated action and emit the cumulative count at each occurrence.
- CRR/status: emit "No" before the video-so-far evidence is sufficient, then
  "Yes" at the first planned probe where the clue is resolved and at every
  later planned probe. Do not return to "No" after a "Yes".
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
- The question should normally sound like it refers to a completed or earlier
  visual moment. Prefer natural event anchors or past-tense wording when the
  same sentence would otherwise read like a current-frame question.
"""
    if profile == "forward":
        return """
Generation guidance for future/next-event cards:
- Ask about a future visual trigger that becomes clear later in the timeline.
- The answer should become determinable in a short future window and should be concise:
  a visible text/number, short event label, or short state phrase grounded in
  the trigger evidence, not a long description.
- Avoid questions whose answer requires keeping the query open across many unrelated
  later events, because long unresolved questions block the next question.
- The user-facing question must make the wait/trigger or prospective nature
  clear in natural language; do not phrase a future card as an ordinary current
  visual QA item.
"""
    if answer_form == "multiple_choice":
        return """
Generation guidance for current/direct cards:
- Ask about what is visible at the grounding chunk: action, spatial relation, count,
  object relation, OCR, or current intent if it is visually supported.
- Distractors should be visually plausible alternatives from nearby or other chunks,
  not generic random words.
- The question should read as an immediate visual judgment. It may use present
  or progressive wording, visible-in-this-view wording, or a compact event
  anchor, but it should not sound like a memory recall or future-trigger task.
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
        if qtype == "multi_emit":
            parts.extend([
                "- Binary multi-answer questions should make the probe semantics explicit.",
                "  For CRR, ask whether the latest/current evidence is now sufficient.",
                "- Avoid mixing current-status wording with historical 'has happened/yet/by now'",
                "  wording unless the family is explicitly CRR-style sufficiency.",
            ])
        else:
            parts.extend([
                "- Binary questions should make the decision boundary explicit and name",
                "  the concrete event, state transition, or current visual condition.",
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


def _temporal_stance_guidance(planned_slots: List[Dict]) -> str:
    """Natural-language timing guidance for teacher card questions.

    This is intentionally prompt-only: pass3 does not post-process question
    strings into templates. The goal is to make the teacher express the same
    temporal contract that placement already enforces.
    """
    slot_hint = ""
    if planned_slots:
        slot_hint = """
- Planned slots include timing_type, temporal_bucket, support_policy, and
  answer_behavior. Copy those structured fields exactly, and also make the
  question wording agree with them. If the evidence supports the answer but
  the question would sound like the wrong timing bucket, rewrite the question
  naturally rather than changing the slot.
"""
    return f"""
Temporal stance guidance for the user-facing question:
- The timing should be understandable from the question text itself, not only
  from metadata. A human viewer should be able to tell whether the question is
  about the current visible moment, a completed earlier moment, a future
  trigger, a prediction from current cues, or an evolving status probe.
- Do not use a fixed template, fixed prefix, or copied example sentence. Vary
  the grammar and anchors, and keep the question natural and compact.
{slot_hint}- For current_direct/current_visual cards, phrase the question as an
  immediate visual judgment: present/progressive state, visible-in-this-view
  relation, current action, current text, current count, or current intent.
  Avoid unanchored references to earlier/later events.
- For past_visual_recall_candidate, past_unanswerable, or
  historical_visual_recall cards, anchor the question to a completed visual
  moment or earlier event. Use natural past-tense/event-anchor wording such as
  a completed action, a before/after relation, a "while/when that event was
  happening" reference, or an earlier shown object/state. Avoid bare present
  questions like "What color are..." when the support is historical; make the
  earlier moment explicit without mentioning chunks or timestamps.
- For future_delayed/proactive_output cards, the question must clearly ask the
  assistant to wait for a later visual trigger and then emit a short output.
  The future trigger should be visually checkable and distinct from the ask
  moment. Use varied trigger language; do not make all cards start with the
  same word.
- For current_future_prediction/future_current_cue cards, ask for the likely
  next action, imminent state, or expected result from current cues. Do not
  make these cards ask only for a current fact, current OCR text, or current
  object attribute; those belong to current perception families.
- For multi_answer repeated-count cards, make cumulative semantics clear
  through natural "so far/by now/up to this point" style wording. For SSR
  current step-status cards, ask an immediate current Yes/No question. For CRR
  sufficiency/status cards, ask whether the visible evidence is now/yet/by this
  point sufficient to resolve the underlying question, because the answer must
  change from No to Yes as the stream progresses.
- If a candidate question would be equally plausible as current, past, or
  future with no wording change, revise it to include a natural temporal
  anchor. If that cannot be done without becoming awkward or revealing the
  answer, omit the card.
""".strip()


def card_generation_prompt(
    family: str,
    evidence: List[Dict],
    target_n: int = 1,
    planned_slots: Optional[List[Dict]] = None,
) -> str:
    """Build a 397B prompt that produces v2 cards for a single family.

    Returns a JSON-instructed prompt. The teacher is asked to emit a JSON
    list whose entries match the v2 schema.
    """
    rule = FAMILY_RULES[family]
    qtype = QUESTION_TYPE_BY_FAMILY[family]
    allowed_answer_forms = ANSWER_FORM_VARIANTS.get(
        family,
        (str(rule["answer_form"]),),
    )
    answer_form_doc = (
        str(rule["answer_form"])
        if len(allowed_answer_forms) == 1
        else "one of " + ", ".join(allowed_answer_forms)
    )
    answer_form_schema = (
        f'"{allowed_answer_forms[0]}"'
        if len(allowed_answer_forms) == 1
        else " | ".join(f'"{x}"' for x in allowed_answer_forms)
    )

    planned_slots = [dict(slot) for slot in (planned_slots or [])]
    if planned_slots:
        target_n = len(planned_slots)
    evidence_text = _format_evidence_timeline(
        evidence,
        max_chars=50000 if planned_slots else 90000,
    )
    generation_guidance = _generation_guidance(rule, qtype).strip()
    style_guidance = _question_style_guidance(rule, qtype).strip()
    temporal_stance_guidance = _temporal_stance_guidance(planned_slots).strip()
    slot_block = ""
    if planned_slots:
        slot_block = f"""
Planned slots for this family. Produce exactly one card per slot when the
evidence supports it; otherwise omit that slot instead of changing chunks.
Every output card MUST copy the matching slot_id and MUST obey that slot's
question_style, question_way, evidence_type, support_policy, temporal_role,
target_ovo_task, gold_emit chunks, and grounding_frames.
Do not normalize, rename, or replace planned-slot field values with synonyms.

{json.dumps(planned_slots, ensure_ascii=False, indent=2)}

Slot-local allowed evidence. The full timeline above is only background
context and a source of plausible distractors. The correct answer and
grounding_frames for each planned slot MUST come only from that slot's
allowed_chunks below. If those chunks do not support a high-quality card,
omit that slot instead of using another part of the video.

{_format_planned_slot_evidence(evidence, planned_slots)}

Slot intent guidance:
{_format_planned_slot_guidance(planned_slots)}
"""

    options_block = ""
    if "multiple_choice" in allowed_answer_forms:
        options_block = """
  "options": ["A) ...", "B) ...", "..."],                # required only when answer_form == "multiple_choice"; 2-5 plausible options
  "correct_option": "A" | "B" | "C" | "D" | "E",          # required only when answer_form == "multiple_choice"; the gold letter"""

    if qtype == "multi_emit":
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}, ...],   '
                     '# multi_emit: ONE entry per occurrence/event chunk; '
                     '"value" is cumulative count for F5, Yes/No status for F7, '
                     'or "event@N" for PN1')
    else:
        emits_doc = ('"gold_emits": [{"chunk": int, "value": str}],         '
                     '# single_emit: exactly 1 entry; chunk = when answer is determinable')
    family_extra = FAMILY_EXTRA_RULES.get(family, "").strip()
    style_mix_doc = (
        "For this ours-unique family, set question_style to \"ours_unique\"."
        if rule.get("ours_unique")
        else (
            "Use the planned slot's question_style exactly. Without planned slots, "
            "aim for a 65-70% benchmark_core, 20-25% benchmark_variant, and "
            "5-10% ours_unique final mix: for the first strong benchmark-like "
            "card use \"benchmark_core\"; if producing a second card, use "
            "\"benchmark_variant\" with the same task type and answer form but "
            "a different natural question sentence. Do not set \"ours_unique\" "
            "for this family."
        )
    )
    e2_answer_leak_doc = (
        "\n- E2/proactive-output exception: the question is allowed to include "
        "the requested output phrase because the skill is detecting the future "
        "trigger and emitting that phrase exactly once."
        if family == "E2"
        else ""
    )
    if planned_slots:
        task_name_lines = []
        for slot in planned_slots:
            task_family = str(slot.get("task_family") or slot.get("slot_group") or "")
            task_subtype = str(slot.get("task_subtype") or slot.get("slot_subtype") or "")
            timing_type = str(slot.get("timing_type") or slot.get("temporal_bucket") or "")
            readable = str(slot.get("readable_task_name") or " / ".join(
                p for p in (task_family, task_subtype, timing_type) if p
            ))
            task_name_lines.append(
                f"- {slot.get('slot_id', '')}: {readable} "
                f"(legacy_family_id={slot.get('legacy_family_id') or slot.get('family') or family})"
            )
        readable_task_doc = "\n".join(task_name_lines)
    else:
        readable_task_doc = (
            f"- {rule['category']} / {rule['family_name']} "
            f"(legacy_family_id={family})"
        )

    return f"""You are a teacher generating training card(s) from a video's per-chunk evidence.

Readable task names:
{readable_task_doc}
Internal legacy family id: {family}
Legacy rule summary: {rule["intent"]}
Category: {rule["category"]} / {rule["family_name"]}
Answer form: {answer_form_doc}
Question type: {qtype}

Evidence timeline from pass1/pass1b (per-chunk visible_entities + facts + spatial + ocr + state_changes + think):
{evidence_text}

{slot_block}

{generation_guidance}

{style_guidance}

{temporal_stance_guidance}

Produce {target_n} card(s) as a JSON list. Each card schema:
{{
  "slot_id": "...",                                      # required when planned slots are provided; copy from the planned slot
  "family": "{family}",
  "question": "...",                                     # bare natural-language question only
  "question_style": "benchmark_core" | "benchmark_variant" | "ours_unique",
  "question_way": "object_attribute|person_identity_interaction|action_recognition|text_readout|spatial_relation|temporal_order|causal_intent|future_prediction|proactive_output|repeated_count|current_status_probe|evidence_sufficiency_probe|unanswerable_absence|sequential_reference|emotion_context|scene_summary|live_narration|source_discrimination|multimodal_alignment",
  "evidence_type": "object_attribute_visual|person_relation_visual|action_event_visual|text_ocr_visual|spatial_relation_visual|temporal_order_visual|causal_context_visual|future_cue_visual|future_trigger_visual|repeated_event_stream|status_probe_stream|absence_unanswerable|global_context_memory|emotion_context_visual|live_state_change|source_discrimination_visual|multimodal_alignment_visual",
  "answer_form": {answer_form_schema},
  "canonical_answer": "...",                              # the final/correct answer text{options_block}
  {emits_doc}
  "grounding_frames": [int, ...],                        # MINIMAL set of chunk indices needed to verify the answer
  "target_ovo_task": "OCR|ACR|ATR|STU|FPD|OJR|EPM|ASI|HLD|REC|SSR|CRR|GLOBAL|STREAMING_AGENT",
  "temporal_role": "current_visual|current_probe|historical_visual_detail|historical_abstention_check|delayed_clue_resolution|cumulative_count|current_step_status|crr_sufficiency_probe|future_current_cue|future_event_wait|live_narration|global_summary",
  "support_policy": "current_visual|historical_visual_recall|historical_state_memory|future_current_cue|probe_status",
  "legacy_family_id": "{family}",                         # copy from planned slot when provided; old internal id only
  "task_family": "current_perception|past_memory|temporal_reasoning|future|multi_state|global_context",
  "task_subtype": "...",                                  # readable subtype; copy from planned slot when provided
  "timing_type": "...",                                   # current/past/future/multi timing; copy from planned slot when provided
  "readable_task_name": "...",                            # task_family / task_subtype / timing_type
  "slot_group": "current_perception|past_memory|temporal_reasoning|future|multi_state|global_context",
  "slot_subtype": "...",                                # copy from planned slot when provided
  "temporal_bucket": "...",                             # copy from planned slot when provided
  "benchmark_source": "...",                            # copy from planned slot when provided
  "benchmark_task": "...",                              # copy from planned slot when provided
  "answer_behavior": "...",                             # copy from planned slot when provided
  "question_goal": "...",                               # copy from planned slot when provided
  "placement_hint": "...",                              # copy from planned slot when provided
  "recall_eligible": true | false,
  "state_memory_required": true | false
}}

Rules:
- question must NOT contain or paraphrase the answer.
{e2_answer_leak_doc}
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
- When planned slots are provided, do not choose your own evidence position:
  gold_emits chunk list MUST exactly equal the slot answer_chunks; for
  multi_emit, emit at every answer_chunks/probe chunk in the same order.
  grounding_frames must be a non-empty subset of slot support_chunks plus
  answer_chunks. If the slot cannot support a high-quality question, omit that
  card rather than moving the question to a different chunk.
- When planned slots provide question_style, question_way, evidence_type,
  support_policy, temporal_role, target_ovo_task, legacy_family_id,
  task_family, task_subtype, timing_type, readable_task_name, slot_group,
  slot_subtype, temporal_bucket, benchmark_source, benchmark_task,
  answer_behavior, question_goal, or placement_hint, copy them exactly into
  the card and follow their intent.
- For planned CRR1 status-probe slots, the emitted probe values must include
  both "Yes" and "No". If every planned probe would truthfully be only "Yes" or
  only "No", omit that slot rather than producing an all-one-label card.
- For planned F7/SSR slots, emit exactly one Yes/No value at the planned
  answer chunk. Do not create a multi_emit F7 card.
- For planned CRR1 slots, values must be monotonic No...Yes...Yes. Never emit
  "No" after a "Yes". The first planned probe should normally be "No"; the
  first "Yes" should occur only when the resolving clue becomes visible. A
  CRR1 card with all "Yes" values is invalid even if the question wording is
  otherwise natural.
- Treat the family as a reasoning type, then set support_policy precisely:
  current_visual means the answer should be asked while support is within the
  active visual window; historical_visual_recall means a concrete old visual
  detail may need recall if asked after the window; historical_state_memory
  means cumulative/state tracking such as REC/counting, not recall;
  future_current_cue means FPD-style prediction from the current cue;
  probe_status means SSR/CRR-style probe answers.
- Set recall_eligible true only for concrete historical visual facts whose
  evidence can be retrieved as frames. Keep F5/REC, F7/SSR, F6/FPD, PN1, and
  ordinary status/state probes recall_eligible=false.
- If producing multiple cards for this family, make them semantically diverse:
  use different events/chunks and different answer types within the family.
  Prefer one immediately visible/current-style card and one event-anchored
  historical-detail card when the evidence supports both. Do not make near
  duplicates with only different options.
- {style_mix_doc}
- benchmark_core should match OVO-Bench or StreamingBench-style wording and
  answer format for this family. benchmark_variant must keep the same evidence
  type and answer format, but use a different user phrasing pattern so the
  model does not overfit benchmark templates. ours_unique is a style/source
  label only; do not use it as a current/past/future timing label.
- Set question_way/evidence_type more specifically than the family. Match
  these benchmark question forms when evidence supports them:
  current perception: object_attribute, action_recognition, text_readout,
  spatial_relation, causal_intent, emotion_context; prior/history:
  person_identity_interaction, temporal_order, causal_intent,
  object_attribute, spatial_relation, unanswerable_absence; active state:
  repeated_count, current_status_probe, evidence_sufficiency_probe;
  proactive output: proactive_output; streaming sequential reference:
  sequential_reference only when there is a stable earlier referent; source
  discrimination: source_discrimination; multimodal consistency or
  contradiction: multimodal_alignment.
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
- For descriptive: canonical_answer is one short grounded sentence, preferably
  no more than 20 words.
- If multiple answer forms are allowed and you produce more than one card,
  include at least one non-multiple-choice card when the evidence supports a
  concise literal or descriptive answer. Keep MC cards benchmark-like; keep
  non-MCQ cards natural and answerable without options.
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
# JSON parsing helpers
# ---------------------------------------------------------------------------


_CHUNK_INT_RE = re.compile(r"-?\d+")


def _coerce_chunk_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        chunk = int(text)
        return chunk if chunk >= 0 else None
    except ValueError:
        pass
    match = _CHUNK_INT_RE.search(text)
    if not match:
        return None
    chunk = int(match.group(0))
    return chunk if chunk >= 0 else None


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
                chunk = _coerce_chunk_int(e["chunk"])
                if chunk is None:
                    continue
                norm_emits.append({"chunk": chunk, "value": str(e["value"])})
        if not norm_emits:
            continue
        c["gold_emits"] = norm_emits
        grounding_frames = []
        for g in c.get("grounding_frames") or []:
            chunk = _coerce_chunk_int(g)
            if chunk is not None:
                grounding_frames.append(chunk)
        c["grounding_frames"] = grounding_frames
        c.setdefault("canonical_answer",
                     norm_emits[-1]["value"] if norm_emits else "")
        c["question_type"] = QUESTION_TYPE_BY_FAMILY.get(family, "single_emit")
        c.update(family_taxonomy(family))
        valid.append(c)
    return valid
