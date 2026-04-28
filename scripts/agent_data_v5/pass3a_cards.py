"""
Pass 3-A: Task Card Generation

Generates question cards from teacher evidence.
Each card defines WHAT to ask (not WHEN or HOW to act).

Two steps:
1. classify_chunks: structural filtering (pure program)
2. generate_cards: per-family 397B calls

Output: task_cards/{video_id}.json
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    AGENT_CHUNK_SEC,
    TASK_CARDS_DIR,
    PASS_CONFIG,
    VISUAL_WINDOW_CHUNKS,
)

logger = logging.getLogger(__name__)

# Family targets per video.
# v9.0: F5/F6/N1 added for OVOBench REC/FPD/HLD coverage.
# v9.3: F5/F6 bumped 2→3 (under-yielded in batch1: 4 / 165 cards across 311 videos).
# v9.4: 4 reasoning families added (CR1 causal, CR2 order, CR3 intent,
#       CR4 compositional). These exercise multi-chunk reasoning that the
#       Layer-0/1 perceptual families don't — specifically targeting OVO
#       CRR/ASI/SSR/EPM tasks where batch1 had near-zero coverage.
FAMILY_TARGETS = {
    # v9.5: 22 families. Post-verify mean ~18-20 cards/video.
    "F1": 2, "F2": 3, "F3": 2, "F4": 2,
    "E1": 2, "E2": 2, "P1": 2, "C1": 2,
    "R1": 1, "S1": 2, "M1": 1,
    "F5": 2,  # repetition counting (OVO REC) — force-attempt
    "F6": 2,  # future prediction (OVO FPD) — force-attempt
    "F7": 2,  # step-progress binary multi-probe (OVO SSR) — force-attempt
    "N1": 2,  # hallucination detection (OVO HLD) — force-attempt
    "CR1": 2, # causal why (state_change → cause attribution)
    "CR2": 2, # temporal ordering (3 distinguishable events)
    "CR3": 1, # goal / intent inference
    "CR4": 2, # compositional multi-observation (AND/OR)
    "CR5": 2, # clue-delayed descriptive multi-probe (OVO CRR)
    "CR6": 2, # feasibility / plausibility (STAR-Feas) — v9.5 new
    "CR7": 2, # object permanence after occlusion (PerceptionTest) — v9.5 new
}

# Families that MUST be attempted on every video, even when classify_chunks
# returns zero chunks for them. v9.1 audit found F5=4 cards across 312 videos
# because most videos lack a 3-chunk same-action run; here we let the teacher
# look at the whole video and decide whether it can construct a question.
# Without this, OVOBench REC/FPD/HLD coverage is structurally absent.
# v9.4: CR1-4 added — reasoning families need teacher to scan the whole video
# and decide if multi-chunk reasoning is possible; structural classify_chunks
# would miss many candidates.
FAMILY_FORCE_ATTEMPT = {
    "F5", "F6", "F7", "N1", "F3", "E2", "S1",
    "CR1", "CR2", "CR3", "CR4", "CR5", "CR6", "CR7",
}

# Retention class derived from family (not from 397B).
# v9.4: CR1/CR2/CR4 = "high" (cross-chunk cause/order/composition must be
# explicitly retained across compression); CR3 = "medium" (goal is summary-
# level, often survives compression as gist).
RETENTION_CLASS = {
    "F1": "low", "F2": "low", "F3": "low",
    "F4": "medium", "P1": "medium", "E2": "medium",
    "C1": "medium", "R1": "medium",
    "E1": "high", "S1": "high", "M1": "high",
    "F5": "low",      # exact count must persist
    "F6": "medium",   # process-aware
    "F7": "low",      # per-step state — must persist exactly
    "N1": "low",      # specific entity absence
    "CR5": "high",    # clue-delayed event — both ask context + clue must be retained
    "CR6": "medium",  # feasibility — current scene state suffices, gist survives compress
    "CR7": "high",    # object permanence — pre-occlusion location must survive compress
    "CR1": "high",    # cause needs to be retained from earlier chunk
    "CR2": "high",    # all 3 ordered events must be retained
    "CR3": "medium",  # goal is gist-level, survives compression
    "CR4": "high",    # both/all observations must be retained
}

# Families that need the "absence" verification path: standard verify checks
# whether the answer is supported by support_chunks; these need to confirm
# that the WRONG options (distractors) are absent from the entire video.
# v9.3: N1 is now multiple_choice (was binary "No"); the set name is kept for
# backward-compat in pass3a:_verify_one_card but no longer implies the answer
# is "No". GRPO's silent_quality HLD branch (gt=="no" and form=="binary") is
# now dead code — left in place; harmless.
NEGATIVE_FAMILIES = {"N1"}

# Shared output schema appended to every family prompt.
# Solves: (1) inconsistent field names across families,
#          (2) uncontrolled canonical_answer format,
#          (3) entity_id leaking into question text,
#          (4) v9.3: MC distractors must be plausible (same video, same fact-type).
_OUTPUT_SCHEMA = """

Output a JSON array. Each element MUST have exactly these fields:
{{
  "question": "...",
  "canonical_answer": "...",
  "answer_form": "binary|multiple_choice|number|short_exact|descriptive",
  "support_chunks": [chunk_idx, ...],
  "visibility_type": "persistent|transient"
}}

canonical_answer format rules (STRICT — these go directly into eval matching):
- binary: exactly "Yes" or "No" (English, capitalized)
- multiple_choice: exactly one letter "A", "B", "C", or "D" — nothing else
- number: digits only, no units, no words (e.g. "3" not "3 times" not "three")
- short_exact: 1-5 English words, no articles, lowercase unless a proper noun
- descriptive: 1-3 sentences

Entity reference rules:
- ALWAYS refer to entities by visual appearance ("person wearing red apron"),
  NEVER by ID ("person_1") — the model cannot see IDs at inference time.

Multiple-choice rules (MUST FOLLOW for any answer_form="multiple_choice"):
- Embed all four options inline: "...question? A. <opt> B. <opt> C. <opt> D. <opt>"
- The four options must be MUTUALLY EXCLUSIVE and SAME-TYPE (all colors, all
  counts, all entities — never mix types).
- Distractors (the 3 wrong options) MUST come from THIS video's evidence:
  attributes/entities/actions visible in OTHER chunks. Never invent wording
  the video does not reflect — the eval splits chunks per OVO and a fabricated
  distractor that contradicts a non-target chunk leaks the answer.
- Randomize the position of the correct answer across A/B/C/D — do NOT bias
  toward A. Across one video's cards, all four positions should appear.
- canonical_answer is the LETTER only; do not include the option text."""

# Per-family prompt templates
FAMILY_PROMPTS = {
    "F1": """Based on the following video chunks containing OCR text or numbers,
generate {n} questions about precise values (price, count, text on screen).
Aim for a 2:1 mix of multiple_choice : number across {n} cards (e.g. for n=3,
two MC + one number; for n=2, one of each).
- multiple_choice: 4 options drawn from numbers/texts visible elsewhere in this
  video. canonical_answer is the LETTER only.
- number: digits only, no units.
- Avoid "binary" — OVO OCR is multi-way.
visibility_type: "transient" for momentary values, "persistent" for always-visible text.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F2": """Based on the following video chunks, generate {n} questions about
visual attributes (color, material, shape, clothing).
answer_form: multiple_choice ONLY (4 same-type options drawn from this video's
attributes — e.g. all colors, or all clothing items). Do NOT use binary.
canonical_answer is the LETTER.
visibility_type: "persistent" for always-visible attributes, "transient" for brief appearances.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F3": """Generate {n} questions about counts/quantities from these chunks.
answer_form: number (digits only, no units).
visibility_type: usually "transient" (counts change).
Question format: "How many X are visible?" — the count must be unambiguously
resolvable from a single chunk's evidence.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F4": """Generate {n} questions about spatial relationships from these chunks.
answer_form: multiple_choice (4 options of the SAME relation type — e.g. all
positions: A. left B. right C. behind D. in front; or all locations: A. on counter
B. in bowl C. on stove D. in sink). Distractors must be locations/positions that
appear in OTHER chunks of this video. Do NOT use binary.
visibility_type: "persistent" for stable layouts, "transient" for moving objects.

{evidence}
""" + _OUTPUT_SCHEMA,

    "E1": """Generate {n} questions about the action a person is currently performing.
Aim for a 2:1 mix of multiple_choice : short_exact across {n} cards.
- multiple_choice: 4 options of action verbs that appear in DIFFERENT chunks
  of this video (e.g. A. stirring B. chopping C. pouring D. wiping). The
  correct option is the action visible in support_chunks; distractors are
  actions visible at OTHER times.
- short_exact: 1-3 word verb phrase ("chopping onions", "pouring oil").
- Avoid "binary" (Yes/No) — OVO ACR is multi-way.
visibility_type: "transient" (actions change).

{evidence}
""" + _OUTPUT_SCHEMA,

    "E2": """Generate {n} event-watch / state-change questions. Aim for a balanced split:
- ONE event_watch question, answer_form="binary": "Has X started yet?" /
  "Is the bread already in the oven?" — answered "No" before the event chunk
  and "Yes" after. canonical_answer reflects the state AT support_chunks.
- THE REST as multiple_choice (clue-reveal / CRR): "What just changed?" with
  4 options drawn from state_changes across this video; correct option = the
  state_change at support_chunks.
canonical_answer for MC is the LETTER.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "P1": """Generate {n} questions about procedure / step order.
answer_form: multiple_choice ONLY (4 options drawn from steps actually
performed in this video, in different chunks). Examples:
- "Which step is the chef performing now? A. mixing B. baking C. plating D. cleaning"
- "What step came IMMEDIATELY before? A. ... B. ... C. ... D. ..."
canonical_answer is the LETTER.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "C1": """Generate {n} comparison questions: how has something changed over time?
answer_form: multiple_choice (4 options describing different possible state
transitions; correct option = the actual change between earlier and later
support_chunks). Examples:
- "How has the pan's contents changed since the start? A. emptied B. now contains
  meat C. now contains vegetables D. now contains liquid"
canonical_answer is the LETTER.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "R1": """Generate {n} re-identification questions about whether a previously seen
entity is still on screen. answer_form: multiple_choice (4 options describing
distinct entities visible in DIFFERENT chunks of this video; correct option
= the entity actually re-identified at support_chunks). Examples:
- "Which of the previously seen people is back on screen? A. <person in red apron>
  B. <person in blue cap> C. <person in white shirt> D. <person in green hoodie>"
Describe each entity by appearance, never by ID. canonical_answer is the LETTER.
visibility_type: "transient".

{evidence}
""" + _OUTPUT_SCHEMA,

    "S1": """Generate {n} descriptive questions about the scene.
Question format: "Describe what is happening" / "What entities are present and what are they doing?"
answer_form: descriptive.
visibility_type: "persistent".

ANSWER FORMAT (strict): canonical_answer must be 30-80 words containing
3-5 SPECIFIC observations (entity descriptions + actions + spatial layout).
DO NOT write meta-commentary like "the video shows" or "we can see".
Each observation must be a concrete visual fact, not interpretation.

{evidence}
""" + _OUTPUT_SCHEMA,

    "M1": """Based on this full video summary, generate {n} questions suitable
for continuous commentary (e.g., "Describe each step as it happens").
answer_form: descriptive.
visibility_type: "transient".

ANSWER FORMAT (strict): canonical_answer must be 30-80 words describing
3-5 distinct moments/steps with timestamps when visible. Format like:
"At [t1], X happens; at [t2], Y begins; ..."
DO NOT write meta narration. Each clause must reference a concrete visible event.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F5": """Based on the following video chunks, generate UP TO {n} questions about
ACTION REPETITION COUNT (OVOBench REC).

The chunks below come from the SAME video. Your job is to SCAN them and find
any action / event that visibly recurs. Recurrence is BROAD — count any of:
  • a person doing the same gesture/movement on multiple distinct occasions
    (stirring, chopping, wiping, knocking, pouring, dipping)
  • an object being acted on the same way repeatedly (egg cracked, dough
    folded, ingredient added)
  • a recurring step in a procedure (each tray placed in oven, each guest
    served, each lap around a track)
  • a recurring visual event (text overlay flashing, scene cut to same
    location, same person re-entering frame)

The repetitions need NOT be in adjacent chunks; they can be spaced across the
timeline. TWO confirmed occurrences IS enough — that gives count=2.

Question format: "How many times did the person {{verb}}?" or
"How many {{action}} occurrences appear?"
answer_form: number (digits only, e.g. "3").
visibility_type: "transient" (count is only complete after the last repetition).
support_chunks MUST list EVERY chunk where an occurrence is observed (≥2).

CRITICAL — output an EMPTY JSON array `[]` ONLY IF you cannot find ANY
recurring action/event across the entire video. Most procedural / cooking /
sports / tutorial videos have at least one recurring action — TRY HARD before
giving up. If two valid candidates exist, output two cards.

{evidence}
""" + _OUTPUT_SCHEMA,

    "F6": """Based on the following video chunks, generate UP TO {n} questions about
FUTURE PREDICTION (OVOBench FPD).

The chunks below come from the SAME video, ordered by time. SCAN them for
any moment where an in-progress process strongly implies what happens next
(e.g. "person grabs knife and a tomato" → next step is cutting; "pours batter
into pan on stove" → next step is cooking/setting). The predicted event must
be observable in a LATER chunk so we can verify the gold answer.

Question format: "What will the person do next?" or "What is about to happen?"
Prefer answer_form: multiple_choice (4 plausible options, one is the actual
continuation; distractors are plausible alternatives from the same domain —
other steps in the procedure or visually adjacent objects).
visibility_type: "transient".
support_chunks: the chunks BEFORE the predicted event (the setup), NOT the
chunks where the event actually happens.

CRITICAL — output an EMPTY JSON array `[]` ONLY IF no chunk pair in the
evidence shows a setup → continuation relationship you are confident about.
If the continuation is genuinely ambiguous, SKIP that case — do NOT fabricate.
A wrong future-prediction gold answer poisons training. Quality > quantity.
But DO scan thoroughly: a typical 2-3 minute procedural video has 1-3 valid
prediction points.

{evidence}
""" + _OUTPUT_SCHEMA,

    "N1": """Based on the following video chunks, generate {n} HALLUCINATION-DETECTION
questions (OVOBench HLD).

OVO HLD format (verified against ovo_bench_new.json: 186/186 samples):
- The QUESTION asks about something the agent CANNOT actually answer from
  the visible evidence (the asked-about entity / action / state is NOT
  shown in any chunk above).
- The four MC options consist of:
    * THREE plausible-but-absent answers (entities / actions / attributes
      that match the video genre but DO NOT appear in the evidence above).
    * ONE option that is literally the string "Unable to answer".
- The CORRECT answer is "Unable to answer" — the agent must recognize
  that the question is unanswerable from the observed video and refuse.
- canonical_answer: the LETTER (A/B/C/D) pointing to "Unable to answer".

Examples (OVO real samples):
  Q: "Where was the tray before I removed it from the oven?"
     A. table  B. shelve  C. Unable to answer  D. rack       gt = "C"
  Q: "What did I put in the black dustbin?"
     A. empty water bottles  B. Unable to answer
     C. old newspapers  D. food scraps                       gt = "B"
  Q: "Where is the small grey keg?"
     A. Unable to answer  B. floor  C. shelf  D. tool box    gt = "A"

Procedure:
1. Pick an entity / event / location that is plausibly part of the video's
   GENRE (cooking, sports, tutorial...) but NOT visible in any chunk above.
2. Phrase a "Where / What / Who" question about it.
3. Make 3 distractor options that are also plausible-genre answers but
   NONE of which appear in the evidence (so even those are wrong).
4. Add "Unable to answer" as the 4th option.
5. Place "Unable to answer" at a randomized A/B/C/D position; canonical_answer
   is that letter.

Question format: "Where/What/Who/When ...?" — must be a content question
the model would normally try to answer. Never phrase as "Is X present?"
(binary collapses the format).

CRITICAL — none of the 4 options (including the 3 distractors) should
actually appear in the evidence above. If you can't construct a question
where all real-content options are absent, output `[]`.

visibility_type: "persistent" (the absence/inability holds throughout).
support_chunks: pick 2-3 chunks that show the SCENE CONTEXT (the genre /
setting that makes the distractors plausible), not where the answer is
(it isn't visible anywhere).

{evidence}
""" + _OUTPUT_SCHEMA,

    # ─── v9.4 reasoning families ──────────────────────────────────────────
    "CR1": """Based on the following video chunks, generate UP TO {n} CAUSAL-WHY
questions (OVO CRR-style reasoning).

A causal-why card has the form: an OUTCOME observable at chunk T_effect, whose
CAUSE was visible at an earlier chunk T_cause < T_effect. The agent must
remember the cause and use it to explain the effect.

Procedure to construct one card:
1. Find a chunk T_effect where a state_change or notable outcome occurs
   (e.g. "pan starts smoking", "dough rises", "mixture darkens", "person
   slips", "light turns red").
2. Look BACKWARD up to ~10 chunks for an action / event that plausibly
   caused it (e.g. "oil heated for 2 min", "yeast added", "heat increased",
   "floor wet", "switch flipped").
3. Verify both cause and effect are visible in the evidence above.

Question format: "Why <effect>?" or "What caused <effect>?"
answer_form: multiple_choice — 4 options:
  - 1 correct cause (visible at T_cause)
  - 3 plausible distractors drawn from OTHER actions/events visible elsewhere
    in this video (NOT invented). Each distractor must be a real
    same-domain action that happened in a different chunk.
canonical_answer: the LETTER of the correct cause.
support_chunks: [T_cause, T_effect] — both required.
visibility_type: "transient".

Bad example (don't do): cause invented from common sense ("oil is hot")
when no chunk shows oil heating.
Good example: chunk 12 shows chef adding water to flour; chunk 18 shows
dough becoming sticky → "Why is the dough sticky now? A. too much water added
B. yeast was activated C. oil was poured D. chef kneaded too long" answer="A".

CRITICAL — output an EMPTY JSON array `[]` ONLY IF you cannot find any
genuine cause→effect pair in the evidence. If you find one, generate a
card; if two, generate two.

{evidence}
""" + _OUTPUT_SCHEMA,

    "CR2": """Based on the following video chunks, generate UP TO {n} TEMPORAL
ORDERING questions (OVO ASI / SSR style).

A temporal-ordering card asks the agent to reproduce the order in which 3
distinguishable events occurred, AFTER all 3 have been observed.

Procedure:
1. Pick THREE events from DIFFERENT chunks that are well-separated in time
   (≥5 chunks apart between consecutive events). Each event must be
   uniquely describable (different verb / different object / different
   entity) so the 3 cannot be confused with each other.
2. Phrase each event as a short clause ("cracked egg", "poured batter",
   "added salt", "lit stove", "opened oven door").

Question format:
  "What was the order of these three events?
   A. <ord_1> → <ord_2> → <ord_3>
   B. <perm_b>
   C. <perm_c>
   D. <perm_d>"

answer_form: multiple_choice. canonical_answer: the LETTER of the correct
permutation. The 3 wrong options are 3 wrong permutations of the same 3
events (do NOT include irrelevant events as distractors).
support_chunks: [chunk_event_1, chunk_event_2, chunk_event_3] in order.
visibility_type: "transient".

CRITICAL — the 3 events must each be UNAMBIGUOUSLY identifiable in the chunks
listed; otherwise the question has no ground truth. If you cannot find 3
clearly distinct, well-separated events, output `[]`.

{evidence}
""" + _OUTPUT_SCHEMA,

    "CR3": """Based on the following video chunks (early-to-mid portion of the
video), generate UP TO {n} GOAL/INTENT questions.

The card asks what overall goal the person in the video is working toward,
inferred from a sequence of preparatory actions. Examples:
- Cooking: "What dish is being prepared?" (omelette / pancakes / soufflé / french toast)
- Sports: "What sport is the person training for?" (boxing / running / cycling / swimming)
- DIY: "What is the person assembling?" (chair / table / bookshelf / cabinet)

answer_form: multiple_choice — 4 options:
  - 1 correct goal (matches what the chunks reveal)
  - 3 plausible goals from the SAME GENRE that share early-stage actions
    with the correct goal (e.g. all four are breakfast dishes that involve
    cracking eggs; all four are leg exercises). Distractors must be
    semantically close so the question requires actually integrating the
    full action sequence — not just genre identification.
canonical_answer: the LETTER.
support_chunks: 3-5 chunks from the evidence that together reveal the goal
(typically actions / ingredients / tools visible across the early sequence).
visibility_type: "persistent" (the goal, once inferable, holds for the rest
of the video).

CRITICAL — if the chunks don't yet reveal a clear goal (too early /
ambiguous), output `[]`. Do NOT guess.

{evidence}
""" + _OUTPUT_SCHEMA,

    "CR4": """Based on the following video chunks, generate UP TO {n}
COMPOSITIONAL questions that REQUIRE COMBINING ≥2 SEPARATE OBSERVATIONS to
answer correctly.

A compositional card has 2 (or 3) sub-conditions that are each established
in DIFFERENT chunks. The agent cannot answer correctly by looking at one
chunk; it must remember both observations and combine them with AND / OR /
NEITHER logic.

Procedure:
1. Pick TWO chunks A and B (separated by ≥4 chunks) that establish two
   distinct facts: e.g. "chef adds onions" at chunk 8, "chef adds tomatoes"
   at chunk 22.
2. Phrase a question that asks about the conjunction.

Question format (one of):
  - AND: "Did the chef use BOTH X AND Y?"
    A. only X used  B. only Y used  C. both X and Y used  D. neither
  - WHICH-PAIR: "Which two ingredients were both added?"
    (4 options each listing a pair; correct one names X and Y)

answer_form: multiple_choice. canonical_answer: the LETTER.
support_chunks: [chunk_A, chunk_B] (or up to 3) — ALL chunks required to
answer must be listed.
visibility_type: "transient".

CRITICAL — if the answer can be obtained from ANY single chunk alone, the
question is NOT compositional and is INVALID. Verify the dependency: both
sub-observations must be needed to distinguish the correct option from the
distractors. If you cannot construct such a question, output `[]`.

{evidence}
""" + _OUTPUT_SCHEMA,

    # v9.5 — OVO SSR alignment. Multi-probe binary "Has step X happened yet?"
    # Generated cards have an EXTRA field `step_chunk` marking when the step
    # actually completes (used by pass3b multi_response to flip No→Yes at the
    # right moment). canonical_answer here is the FINAL state ("Yes") since
    # pass3b derives per-probe answers from step_chunk.
    "F7": """Based on the following video chunks, generate UP TO {n} STEP-PROGRESS
questions aligned with OVOBench SSR.

The video shows a procedural / instructional / multi-step activity (cooking,
DIY, exercise, tutorial). Identify {n} DISTINCT named steps that complete at
identifiable chunks (e.g., "pour the egg", "flip the steak", "tighten the bolt",
"start the engine").

For each step, generate a question of the form:
  "Has the step '<step description>' been done by now?"
canonical_answer: "Yes" (the FINAL state — the step is observed completing
                  at step_chunk in the video; pass3b will flip the answer
                  to "No" at chunks before step_chunk during multi-probe
                  rendering).
answer_form: "binary".
visibility_type: "transient" (the answer flips from No→Yes at step_chunk).

REQUIRED extra fields per card (in addition to the standard schema):
  "step_chunk": the chunk_idx where the step is OBSERVED COMPLETING. The
                multi-probe engine uses this to render "No" responses at
                chunks < step_chunk and "Yes" at chunks >= step_chunk.
  "step_label": short human-readable label for the step (≤6 words, no
                trailing period).

support_chunks: [step_chunk] — single chunk where the step completion is
visible (the model needs this for evidence retention; usually identical to
step_chunk).

CRITICAL — pick steps that are visually unambiguous at step_chunk. If a step
is gradual or hard to localize (e.g., "the dough rises"), SKIP it; OVO SSR
expects clean before/after binary judgments.

If fewer than {n} clean steps exist, output fewer cards. Output `[]` only if
the video has no procedural structure at all.

{evidence}
""" + _OUTPUT_SCHEMA,

    # v9.5 — STAR-Feasibility alignment. Plausibility judgment over what
    # the agent has SO FAR observed (not future).
    "CR6": """Based on the following video chunks, generate UP TO {n}
FEASIBILITY questions aligned with STAR-Feas.

Find moments where given the visible context (entities, tools, materials,
on-screen state) it is possible to ask whether some hypothetical action
COULD reasonably be performed. The answer must be derivable from
visible evidence — never from world-knowledge guessing.

Question format: "Given what is visible so far, could the {{actor}}
{{hypothetical action}}?" — present 4 options where exactly one is
plausibly supported by the evidence.

Examples:
  - "Could the chef bake cookies right now? A. Yes — flour, sugar, eggs,
     and an oven are all present  B. No — no flour visible  C. No — no
     oven visible  D. No — no eggs visible"
  - "Could the cyclist start riding now? A. Yes — both wheels mounted,
     handlebars attached  B. No — chain not connected  C. No — front
     wheel still off  D. No — saddle missing"

answer_form: "multiple_choice".
canonical_answer: the LETTER (A/B/C/D).
visibility_type: "persistent" (the visible state at ask_chunk supports
the judgment).

CRITICAL — the option set must include exactly one Yes-with-evidence
and three No-with-different-missing-prerequisite. Do NOT mix Yes
options or use vague "maybe". Distractor No options must each cite a
DIFFERENT specific missing item that is plausibly required (so the
student must check ALL prerequisites, not just one).

If the video shows no scenario where feasibility is testable from
visible evidence, output `[]`.

{evidence}
""" + _OUTPUT_SCHEMA,

    # v9.5 — PerceptionTest object-permanence alignment. Track an entity
    # through an occlusion / camera-cutaway and ask its post-event location
    # or state.
    "CR7": """Based on the following video chunks, generate UP TO {n}
OBJECT-PERMANENCE questions aligned with PerceptionTest.

Find a moment in the video where:
  - At chunk A, an entity is visible at a specific position (e.g.,
    "ball on left side of table", "remote on the couch").
  - Between chunks A and B, the entity is occluded — covered by another
    object, the camera cuts to a different angle/location, or the entity
    moves out of frame.
  - At chunk B (after the occlusion), the entity's location/state CAN be
    inferred from indirect evidence (e.g., a hand still holding it,
    a bulge under a cloth, the box now closed with it inside).

Generate question:
  "After {{occlusion event}}, where is the {{entity}} now?" with 4
  position/state options.

Examples:
  - "After the magician covered the ball with the cup, where is the
     ball? A. Under the cup  B. In his hand  C. On the table  D. Gone"
  - "After the camera turned back to the kitchen, what state is the
     bread in? A. Still in the oven  B. On the cooling rack  C. On the
     plate  D. Still raw on the counter"

answer_form: "multiple_choice".
canonical_answer: the LETTER (A/B/C/D).
visibility_type: "transient" (state changes through occlusion).

REQUIRED extra fields:
  "occlusion_chunk": chunk_idx where the occlusion / cutaway begins.
  "resolve_chunk":   chunk_idx where the post-occlusion state is shown.

support_chunks: [pre-occlusion chunk, resolve_chunk] — the answer needs
both the BEFORE state and the AFTER inference signal.

CRITICAL — the entity's post-occlusion location/state MUST be
recoverable from on-screen evidence. If the model would need pure
world-knowledge guessing (no visible cue at all), the question is
INVALID — output a different one.

If no clear occlusion-and-resolve event exists in this video,
output `[]`.

{evidence}
""" + _OUTPUT_SCHEMA,

    # v9.5 — OVO CRR alignment. Multi-probe descriptive answer with clue
    # delay. The model must answer "I don't know yet" / silent at probes
    # before clue_chunk and a free-text descriptive answer at probes from
    # clue_chunk onward. canonical_answer is the post-clue descriptive answer.
    "CR5": """Based on the following video chunks, generate UP TO {n} CLUE-DELAYED
descriptive questions aligned with OVOBench CRR.

Find a moment in the video where:
  - At chunk A (the "ask" moment), an entity, situation, or unresolved
    interaction sets up a question that someone would naturally wonder
    about (e.g., "two people are talking — what do they do next?",
    "the woman walks toward the car — what does she do to the car?",
    "the player stares at the puzzle — what does she try to do?").
  - At chunk B > A (the "clue" moment), the answer becomes visible
    (e.g., "she pushes him into the water", "they walk to a dark
    corridor", "she rotates the puzzle piece").

For each such (A, B) pair, generate a card:
  question: a natural-language descriptive question about the unresolved
            event at chunk A. The wording should make sense AT chunk A
            without requiring B's information.
  canonical_answer: a 5-15 word free-text DESCRIPTIVE answer that is only
                    derivable from chunk B and onward. NO MCQ letters.
  answer_form: "descriptive".
  visibility_type: "transient".

REQUIRED extra fields:
  "ask_chunk": A — chunk where the question is naturally asked.
  "clue_chunk": B — chunk where the answer becomes visible (B > A).

support_chunks: [B] — single chunk where the resolution is visible.

CRITICAL — the question MUST be ambiguous at chunk A (no choice between
discrete options) and MUST become unambiguous at chunk B. Skip any case
where the answer is already visible at chunk A.

If the video has no such ask→clue structure, output `[]`.

{evidence}
""" + _OUTPUT_SCHEMA,
}


# ---------------------------------------------------------------------------
# Step 1: Structural chunk classification
# ---------------------------------------------------------------------------


def _desc_overlap(desc_a: str, desc_b: str) -> float:
    """Word-level overlap ratio between two entity descriptions."""
    words_a = set(re.findall(r'[a-zA-Z]{2,}', desc_a.lower()))
    words_b = set(re.findall(r'[a-zA-Z]{2,}', desc_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


_ACTION_HEAD_RE = re.compile(r"^([a-zA-Z]+)")


def _action_verb_lemma(action: str) -> str:
    """Extract the head verb from an action phrase (lemma-ish, no NLP dep).

    "stirring the pot" → "stir"; "stirs vigorously" → "stir"; "stirred" → "stir".
    Used to compare actions across chunks for F5 (repetition) without being
    tripped by tense / objects / adverbs. Handles the doubled-consonant
    variant ("stirring" → "stirr" → "stir") which is common in cooking
    captions. Does NOT handle silent-e drop ("hated" → "hat" not "hate"),
    but as long as both chunks use the same tense the comparison still works.
    """
    if not action:
        return ""
    a = action.lower().strip()
    m = _ACTION_HEAD_RE.match(a)
    if not m:
        return a
    head = m.group(1)
    # Strip a single inflectional suffix, longest first.
    for suf, doubled in (("ing", True), ("ed", True), ("s", False)):
        if head.endswith(suf) and len(head) > len(suf) + 2:
            stem = head[: -len(suf)]
            # "stirring" → "stirr" → "stir": collapse doubled trailing
            # consonant introduced by -ing / -ed gerund formation.
            if (doubled and len(stem) >= 2
                    and stem[-1] == stem[-2]
                    and stem[-1] not in "aeiou"):
                stem = stem[:-1]
            return stem
    return head


def _get_primary_action(cap: Dict) -> str:
    """Extract the primary entity action from a chunk (1-A direct field).

    Returns the lemma-ish head verb so consecutive chunks of "stirring" /
    "stirs" / "stirred" all match for F5 repetition detection.
    """
    for e in cap.get("visible_entities", []):
        action = e.get("action", "")
        if action:
            return _action_verb_lemma(action)
    return ""


def chunk_has_evidence(cap: Dict) -> bool:
    """Whether a 1-A chunk has any usable signal for question generation.

    A chunk passes iff at least one of (visible_entities, atomic_facts,
    ocr, spatial) is non-empty. Silent-empty chunks (json-valid but all
    fields empty — 46% of pre-v9.5 batch1) are filtered out so they
    don't get mistakenly chosen as support_chunks.
    """
    return bool(cap.get("visible_entities") or cap.get("atomic_facts")
                or cap.get("ocr")
                or (cap.get("spatial") and str(cap.get("spatial")).strip()))


def classify_chunks(evidence: List[Dict]) -> Dict[str, List[int]]:
    """Classify chunks by family using structural fields.

    Primary path: 1-A direct fields (ocr, visible_entities, atomic_facts).
    Fallback path for P1/C1/R1: 1-A action/desc fields when 1-B fields
    (state_changes, entity_id) are missing, to reduce false negatives.

    v9.5: silent-empty chunks (parse_success=False or _silent_empty=True
    or all evidence fields empty) are pre-filtered. Without this, ~46% of
    batch1 chunks looked "selectable" but contributed no signal — pass3a
    would pick them as support, then pass4 would later reject the cards.
    """
    evidence = [cap for cap in evidence if chunk_has_evidence(cap)]
    fc = {f: [] for f in FAMILY_TARGETS}

    for cap in evidence:
        idx = cap.get("chunk_idx", 0)
        entities = cap.get("visible_entities", [])
        facts = [f for f in cap.get("atomic_facts", [])
                 if f.get("confidence", 0) >= 0.7]

        has_digit_facts = any(
            re.search(r'\d{2,}|[\$€¥£]\d|\d\s*(?:kg|lb|ml|oz|cm|mm|g)\b', f.get("fact", ""))
            for f in facts
        )

        if cap.get("ocr") or has_digit_facts:
            fc["F1"].append(idx)
        if has_digit_facts:
            fc["F3"].append(idx)

        if entities:
            fc["F2"].append(idx)
        if len(entities) >= 2:
            fc["F4"].append(idx)

        if cap.get("state_changes"):
            fc["E2"].append(idx)

        if len(entities) >= 3:
            fc["S1"].append(idx)

    # E1: subsample — guarantee at least FAMILY_TARGETS["E1"] candidates
    all_chunks = [cap["chunk_idx"] for cap in evidence if cap.get("atomic_facts")]
    target_e1 = FAMILY_TARGETS.get("E1", 3)
    step = max(1, len(all_chunks) // max(target_e1 * 2, 1))
    fc["E1"] = all_chunks[::step]

    # ------------------------------------------------------------------
    # P1: procedure detection
    #   Primary: consecutive state_changes >= 3 (from 1-B)
    #   Fallback: consecutive chunks with distinct primary actions >= 3
    #             (from 1-A visible_entities[].action, direct visual field)
    # ------------------------------------------------------------------
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}

    # Primary path: 1-B state_changes
    consecutive = []
    for cap in evidence:
        if cap.get("state_changes"):
            consecutive.append(cap["chunk_idx"])
        else:
            if len(consecutive) >= 3:
                fc["P1"].extend(consecutive)
            consecutive = []
    if len(consecutive) >= 3:
        fc["P1"].extend(consecutive)

    # Fallback path: 1-A action diversity
    action_run = []
    seen_actions = set()
    for cap in evidence:
        action = _get_primary_action(cap)
        if action and action not in seen_actions:
            action_run.append(cap["chunk_idx"])
            seen_actions.add(action)
        else:
            if len(action_run) >= 3:
                fc["P1"].extend(action_run)
            action_run = []
            seen_actions = {action} if action else set()
            if action:
                action_run = [cap["chunk_idx"]]
            else:
                action_run = []
    if len(action_run) >= 3:
        fc["P1"].extend(action_run)

    # ------------------------------------------------------------------
    # C1/R1: cross-chunk entity tracking
    #   Primary: entity_id from 1-B alignment
    #   Fallback: desc word-overlap from 1-A (direct visual field)
    # ------------------------------------------------------------------

    # Primary path: 1-B entity_id
    entity_appearances = {}
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            eid = e.get("id", "")
            if eid and eid != "unknown":
                entity_appearances.setdefault(eid, []).append(cap["chunk_idx"])

    for eid, chunks in entity_appearances.items():
        # C1: same entity in different chunks with state_change
        state_chunks = [c for c in chunks
                        if ev_by_idx.get(c, {}).get("state_changes")]
        if len(state_chunks) >= 2:
            fc["C1"].extend(state_chunks[-2:])
        # R1: same entity reappears after gap >= 5
        for i in range(1, len(chunks)):
            if chunks[i] - chunks[i - 1] >= 5:
                fc["R1"].append(chunks[i])

    # Fallback path: 1-A desc similarity (covers 1-B entity_id misses)
    # Build desc→chunks index from 1-A direct visual field
    desc_chunks = {}  # desc_text -> [(chunk_idx, action)]
    for cap in evidence:
        for e in cap.get("visible_entities", []):
            desc = e.get("desc", "")
            if desc and len(desc) > 5:
                desc_chunks.setdefault(desc, []).append(
                    (cap["chunk_idx"], e.get("action", ""))
                )

    # Match desc pairs with high overlap but different literal desc
    descs = list(desc_chunks.keys())
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            if _desc_overlap(descs[i], descs[j]) < 0.6:
                continue
            chunks_i = desc_chunks[descs[i]]
            chunks_j = desc_chunks[descs[j]]
            all_cidx = sorted(set(c for c, _ in chunks_i + chunks_j))

            # C1 fallback: same-looking entity with different actions
            actions_by_chunk = {}
            for c, a in chunks_i + chunks_j:
                actions_by_chunk.setdefault(c, set()).add(a)
            action_change_chunks = [c for c in all_cidx
                                    if len(actions_by_chunk.get(c, set())) >= 1]
            distinct_actions = set()
            for c in action_change_chunks:
                distinct_actions |= actions_by_chunk[c]
            if len(distinct_actions) >= 2 and len(action_change_chunks) >= 2:
                fc["C1"].extend(action_change_chunks[-2:])

            # R1 fallback: similar desc reappears after gap >= 5
            for k in range(1, len(all_cidx)):
                if all_cidx[k] - all_cidx[k - 1] >= 5:
                    fc["R1"].append(all_cidx[k])

    # ------------------------------------------------------------------
    # F5: action repetition count (OVO REC)
    #   Detect runs of SAME primary_action across >= 3 consecutive chunks.
    #   Distinct from P1 which detects runs of DIFFERENT actions.
    # ------------------------------------------------------------------
    rep_run = []
    rep_action = ""
    for cap in evidence:
        action = _get_primary_action(cap)
        if action and action == rep_action:
            rep_run.append(cap["chunk_idx"])
        else:
            if len(rep_run) >= 3:
                fc["F5"].extend(rep_run)
            rep_run = [cap["chunk_idx"]] if action else []
            rep_action = action
    if len(rep_run) >= 3:
        fc["F5"].extend(rep_run)

    # ------------------------------------------------------------------
    # F6: future prediction (OVO FPD)
    #   Pick chunks that (a) have state_changes or sequential action, AND
    #   (b) have at least 2 chunks of evidence remaining after them
    #   so the predicted continuation is observable.
    # ------------------------------------------------------------------
    if evidence:
        max_idx = max(cap.get("chunk_idx", 0) for cap in evidence)
        for cap in evidence:
            idx = cap.get("chunk_idx", 0)
            if idx > max_idx - 2:
                continue  # need future evidence
            if cap.get("state_changes") or _get_primary_action(cap):
                fc["F6"].append(idx)

    # ------------------------------------------------------------------
    # N1: hallucination negative (OVO HLD)
    #   Pick chunks with rich entity context (>=2 entities) so the teacher
    #   has enough scene grounding to construct plausible-but-absent
    #   negative questions. Distribute across the video timeline.
    # ------------------------------------------------------------------
    n1_candidates = [
        cap["chunk_idx"] for cap in evidence
        if len(cap.get("visible_entities", [])) >= 2
    ]
    n1_target = FAMILY_TARGETS.get("N1", 2) * 3  # 3x oversample for teacher selection
    if n1_candidates:
        step = max(1, len(n1_candidates) // max(n1_target, 1))
        fc["N1"] = n1_candidates[::step][:n1_target]

    # ------------------------------------------------------------------
    # v9.4 — REASONING FAMILIES: CR1 / CR2 / CR3 / CR4
    # These need CROSS-CHUNK context, so we pass each chunk in the
    # candidate set PLUS its preceding context (effect needs cause).
    # All 4 are FAMILY_FORCE_ATTEMPT — even an empty candidate list
    # falls back to whole-video evenly-spaced chunks via generate_cards.
    # ------------------------------------------------------------------

    # CR1 (causal why): chunks with state_changes ARE the effects;
    # include them + all chunks before them so teacher can find causes.
    # We give the teacher up to 10 chunks total: each effect chunk + a
    # preceding window. Cap at 3 effects so the prompt isn't overloaded.
    effect_chunks = [cap["chunk_idx"] for cap in evidence if cap.get("state_changes")]
    if effect_chunks:
        cr1_set = set()
        for ec in effect_chunks[:3]:
            for c in range(max(0, ec - 8), ec + 1):
                cr1_set.add(c)
        # Intersect with chunks that actually exist in evidence
        existing = {cap["chunk_idx"] for cap in evidence}
        fc["CR1"] = sorted(c for c in cr1_set if c in existing)

    # CR2 (temporal ordering): pick high-information chunks well spread
    # across the timeline. We want the teacher to see ≥3 distinguishable
    # events. Score each chunk by (#facts + #state_changes + #entities)
    # and keep up to 8 evenly distributed across the timeline.
    scored = []
    for cap in evidence:
        score = (
            len([f for f in cap.get("atomic_facts", []) if f.get("confidence", 0) >= 0.7])
            + 2 * len(cap.get("state_changes", []))
            + len(cap.get("visible_entities", []))
        )
        if score >= 2:
            scored.append((cap["chunk_idx"], score))
    if len(scored) >= 3:
        # Even-distribute: bin chunks into 8 buckets across the timeline,
        # keep the top-scoring chunk in each bucket.
        ordered = sorted(scored, key=lambda x: x[0])
        if len(ordered) > 8:
            n_bins = 8
            bins: List[List[Tuple[int, int]]] = [[] for _ in range(n_bins)]
            min_c = ordered[0][0]
            max_c = ordered[-1][0]
            span = max(1, max_c - min_c)
            for c, s in ordered:
                b = min(n_bins - 1, (c - min_c) * n_bins // span)
                bins[b].append((c, s))
            picked = []
            for b in bins:
                if b:
                    picked.append(max(b, key=lambda x: x[1])[0])
            fc["CR2"] = sorted(picked)
        else:
            fc["CR2"] = [c for c, _ in ordered]

    # CR3 (goal/intent): give the teacher the FIRST 60-70% of the video.
    # The goal is best inferable from the early action sequence; reading
    # the whole video would make the question trivial (the goal is
    # explicit by the end). Cap at 10 evenly-spaced chunks.
    if evidence:
        all_idx = sorted(cap["chunk_idx"] for cap in evidence)
        cutoff = max(1, int(len(all_idx) * 0.65))
        early = all_idx[:cutoff]
        if early:
            step = max(1, len(early) // 10)
            fc["CR3"] = early[::step][:10]

    # CR4 (compositional): pick chunks where DIFFERENT entities or actions
    # appear, so the teacher can construct AND-style questions across them.
    # Heuristic: chunks whose visible_entities[0].desc differs from the
    # previous selected chunk (forces entity diversity).
    cr4_picks: List[int] = []
    last_desc_set: set = set()
    for cap in sorted(evidence, key=lambda c: c.get("chunk_idx", 0)):
        descs = {e.get("desc", "")[:30] for e in cap.get("visible_entities", []) if e.get("desc")}
        if not descs:
            continue
        # Only pick if new entities relative to last pick
        if descs - last_desc_set:
            cr4_picks.append(cap["chunk_idx"])
            last_desc_set = descs
        if len(cr4_picks) >= 8:
            break
    if len(cr4_picks) >= 2:
        fc["CR4"] = cr4_picks

    # Deduplicate
    for f in fc:
        fc[f] = sorted(set(fc[f]))

    return fc


# ---------------------------------------------------------------------------
# Step 2: Per-family 397B card generation
# ---------------------------------------------------------------------------


def _format_evidence_for_prompt(evidence: List[Dict], chunk_indices: List[int]) -> str:
    """Format selected chunks' evidence into a compact prompt string."""
    ev_by_idx = {cap["chunk_idx"]: cap for cap in evidence}
    lines = []
    for idx in chunk_indices[:10]:  # limit to 10 chunks per call
        cap = ev_by_idx.get(idx)
        if not cap:
            continue
        t = cap.get("time", [idx * AGENT_CHUNK_SEC, (idx + 1) * AGENT_CHUNK_SEC])
        entities = [e.get("desc", "?") for e in cap.get("visible_entities", [])]
        facts = [f["fact"] for f in cap.get("atomic_facts", [])
                 if f.get("confidence", 0) >= 0.7]
        ocr = cap.get("ocr", [])
        sc = cap.get("state_changes", [])
        parts = [f"chunk {idx} [{t[0]}-{t[1]}s]"]
        if entities:
            parts.append(f"entities: {entities}")
        if facts:
            parts.append(f"facts: {facts}")
        if ocr:
            parts.append(f"ocr: {ocr}")
        if sc:
            parts.append(f"changes: {sc}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _parse_cards_response(raw: Optional[str], family: str, video_id: str) -> List[Dict]:
    """Parse 397B response into card dicts."""
    if not raw:
        return []
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Try parse as JSON array
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract array
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning(f"  [{video_id}] 3-A {family}: failed to parse cards")
    return []


async def _generate_family_cards(
    family: str, chunk_list: List[int],
    evidence: List[Dict], client, video_id: str,
) -> List[Dict]:
    """Generate cards for a single family. Called concurrently per family."""
    target_n = FAMILY_TARGETS[family]

    if family == "M1":
        ev_text = _format_evidence_for_prompt(evidence, [cap["chunk_idx"] for cap in evidence])
    else:
        ev_text = _format_evidence_for_prompt(evidence, chunk_list)

    if not ev_text.strip():
        return []

    prompt_template = FAMILY_PROMPTS.get(family)
    if not prompt_template:
        return []

    prompt = prompt_template.format(n=target_n, evidence=ev_text)

    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=PASS_CONFIG.get("pass3a", {}).get("max_tokens", 16384),
        temperature=0.7,
        request_id=f"{video_id}_3a_{family}",
    )

    cards = _parse_cards_response(raw, family, video_id)

    valid = []
    for card in cards:
        if not isinstance(card, dict) or not card.get("question"):
            continue
        card["family"] = family
        card.setdefault("answer_form", "short_exact")
        card.setdefault("visibility_type", "transient")
        if "support_chunks" not in card or not card["support_chunks"]:
            card["support_chunks"] = chunk_list[:1] if chunk_list else [0]
            card["_support_inferred"] = True
            logger.warning(
                f"  [{video_id}] 3-A {family}: "
                f"support_chunks missing, inferred {card['support_chunks']}")
        if isinstance(card["support_chunks"], int):
            card["support_chunks"] = [card["support_chunks"]]
        valid.append(card)
    return valid


async def generate_cards(
    video_id: str,
    evidence: List[Dict],
    client,
) -> List[Dict]:
    """Generate task cards for one video via per-family 397B calls.

    All family calls are independent and run concurrently via asyncio.gather.
    Returns list of TaskCard dicts.
    """
    family_chunks = classify_chunks(evidence)

    # Whole-video fallback chunk list for families in FAMILY_FORCE_ATTEMPT
    # (when their structural classification yields nothing). We sample evenly
    # across the timeline so the teacher sees a representative cross-section.
    # Note: F5/F6 need to *scan* a wide span to find repetition / process
    # setups, so this fallback set caps at the prompt's 10-chunk format limit
    # (see _format_evidence_for_prompt) but spans the whole video.
    all_chunk_idxs = [cap["chunk_idx"] for cap in evidence]
    fallback_chunks: List[int] = []
    if all_chunk_idxs:
        n_total = len(all_chunk_idxs)
        step = max(1, n_total // 10)  # 10 evenly-spaced chunks
        fallback_chunks = sorted(set(all_chunk_idxs[::step]))[:10]

    # Build tasks for families that have candidates OR are force-attempt
    tasks = []
    family_order = []
    for family in FAMILY_TARGETS:
        chunk_list = family_chunks.get(family, [])
        if not chunk_list and family != "M1":
            if family in FAMILY_FORCE_ATTEMPT and fallback_chunks:
                # Use whole-video fallback so teacher can decide if a question
                # is possible (e.g. F5 needs ≥3 same-action chunks somewhere).
                chunk_list = fallback_chunks
            else:
                continue
        tasks.append(_generate_family_cards(family, chunk_list, evidence, client, video_id))
        family_order.append(family)

    # Fire all family calls concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Collect and assign sequential card_ids
    all_cards = []
    card_counter = 0
    for family, result in zip(family_order, results):
        if isinstance(result, Exception):
            logger.error(f"  [{video_id}] 3-A {family}: call failed: {result}")
            continue
        for card in result:
            card_counter += 1
            card["card_id"] = f"{video_id}_{family}_{card_counter:03d}"
            all_cards.append(card)

    family_counts = {f: sum(1 for c in all_cards if c["family"] == f)
                     for f in FAMILY_TARGETS if any(c["family"] == f for c in all_cards)}
    counts_str = ", ".join(f"{f}:{n}" for f, n in family_counts.items())
    logger.info(f"  [{video_id}] 3-A: {len(all_cards)} cards ({counts_str})")

    return all_cards


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def save_cards(video_id: str, cards: List[Dict]):
    TASK_CARDS_DIR.mkdir(parents=True, exist_ok=True)
    path = TASK_CARDS_DIR / f"{video_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cards, f, ensure_ascii=False, indent=2)


def load_cards(video_id: str) -> Optional[List[Dict]]:
    from .cache_version import stage_version_ok
    if not stage_version_ok("3a"):
        return None
    path = TASK_CARDS_DIR / f"{video_id}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_STOP_WORDS = frozenset({
    # Articles / pronouns / prepositions / conjunctions
    "the", "a", "an", "is", "was", "in", "on", "at", "to", "of",
    "and", "or", "it", "yes", "no", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "this", "that",
    "there", "here", "not", "but", "if", "so", "than", "then",
    "just", "about", "up", "out", "its", "his", "her", "my", "your",
    "their", "our", "me", "him", "them", "us", "we", "they",
    "you", "he", "she", "with", "for", "from", "by", "as",
    # Interrogatives / question function words — never appear in thinks
    "what", "which", "who", "whom", "whose", "how", "when", "where",
    "many", "much", "any", "some", "other", "tell", "describe",
    "currently", "now", "still", "yet", "ever", "already",
})


def extract_keywords(text: str) -> List[str]:
    """Extract content keywords from a text string, filtering stop words."""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    seen = set()
    result = []
    for w in words:
        if w not in _STOP_WORDS and len(w) > 1 and w not in seen:
            seen.add(w)
            result.append(w)
    return result


def _extract_mc_choice_text(question: str, answer_letter: str) -> str:
    """Extract the text of the correct MC choice from the question.

    Supports formats:
      "... A.Red B.Blue C.White D.Green"
      "... A. Red B. Blue C. White D. Green"
      "... A) Red B) Blue"
    """
    answer_letter = answer_letter.strip().upper()
    # Build pattern: "A.Red" or "A. Red" or "A) Red", capture until next choice or end
    pattern = rf'(?:^|\s){answer_letter}[\.\)]\s*(.+?)(?:\s+[B-Z][\.\)]|$)'
    m = re.search(pattern, question, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def extract_card_keywords(card: Dict) -> List[str]:
    """Extract discriminative keywords from a card for retention matching.

    The canonical_answer alone is useless for binary ("Yes") and MC ("A").
    This function extracts keywords that would actually appear in a think
    or summary if the student observed the relevant evidence.

    Strategy by answer_form:
      binary:          question keywords (the predicate being judged)
      multiple_choice: question subject + correct choice text
      number:          question keywords + the number itself
      short_exact:     canonical_answer keywords (already informative)
      descriptive:     canonical_answer keywords (already informative)
    """
    answer_form = card.get("answer_form", "short_exact")
    question = card.get("question", "")
    canonical = card.get("canonical_answer", "")

    if answer_form == "binary":
        # "Is the apron red?" → ["apron", "red"]
        return extract_keywords(question)

    elif answer_form == "multiple_choice":
        # "What color? A.Red B.Blue C.White D.Green", answer="A"
        # → question subject ["color"] + correct choice ["red"]
        # Strip choices from question to get the subject part
        q_base = re.split(r'\s+A[\.\)]', question, maxsplit=1)[0]
        q_kw = extract_keywords(q_base)
        choice_text = _extract_mc_choice_text(question, canonical)
        c_kw = extract_keywords(choice_text)
        # Deduplicate preserving order
        seen = set()
        result = []
        for w in q_kw + c_kw:
            if w not in seen:
                seen.add(w)
                result.append(w)
        return result

    elif answer_form == "number":
        # "How many tomatoes were cut?" answer="3"
        # → ["tomatoes", "cut", "3"]
        q_kw = extract_keywords(question)
        num = canonical.strip()
        if num and num not in {kw for kw in q_kw}:
            q_kw.append(num)
        return q_kw

    else:
        # short_exact / descriptive: canonical_answer is already informative
        return extract_keywords(canonical)


# ---------------------------------------------------------------------------
# Card Verification (397B, independent per card, high concurrency)
# ---------------------------------------------------------------------------

VERIFY_CARD_PROMPT = """Verify this video question card against the source evidence.

Evidence chunks:
{evidence}

Card:
- question: "{question}"
- canonical_answer: "{canonical_answer}"
- answer_form: {answer_form}
- support_chunks: {support_chunks}
- visibility_type: {visibility_type}

Check:
1. Is the question answerable from the evidence above?
2. Does the question REQUIRE visual observation to answer? Reject pure common-sense
   questions that could be answered without watching the video (e.g. "Is water wet?").
3. Does the question text leak the answer? (e.g. "The red apron is what color?")
4. Which chunk indices actually contain the answer evidence?
5. Is the answer always visible throughout the video (persistent) or only momentarily (transient)?
6. Is canonical_answer correctly formatted?
   - binary: exactly "Yes" or "No"
   - multiple_choice: exactly one letter "A"/"B"/"C"/"D"
   - number: digits only, no units
   - short_exact: 1-5 English words, no articles

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "persistent|transient", "canonical_answer": "..."}}
If ANY check fails (not answerable, not visual-dependent, or answer leaked), output:
{{"valid": false}}"""

# v9.4 — Reasoning-family verification.
# Standard VERIFY_CARD_PROMPT only checks "is it answerable" — for CR1/CR2/CR4
# this misses the core failure mode: the card may be answerable but doesn't
# actually exercise reasoning (e.g. CR1 with a fabricated cause, CR2 with a
# wrong permutation as gold, CR4 answerable from one chunk alone).
VERIFY_REASONING_PROMPT = """Verify this REASONING question card (family={family}).

Evidence chunks (focused on support_chunks ± 2):
{evidence}

Card:
- family: {family}        # CR1=causal-why, CR2=temporal-ordering, CR4=compositional
- question: "{question}"
- canonical_answer: "{canonical_answer}"  (a single letter A/B/C/D)
- support_chunks: {support_chunks}

Standard checks:
1. Is the question answerable from the evidence? (grounded, not common sense)
2. Does the question text leak the answer?
3. Is canonical_answer exactly one letter A/B/C/D?

Family-specific checks (CRITICAL — primary reason a card should be rejected):

If family == "CR1" (causal-why):
  C1. The question must name a clear EFFECT observable in the evidence.
  C2. The option labeled by canonical_answer must be a CAUSE that is also
      observable in the evidence (in an EARLIER chunk than the effect).
  C3. The cause→effect link must be genuine — the cause must plausibly
      bring about the effect (not just temporally earlier).
  C4. The 3 distractor options must be SAME-DOMAIN actions visible elsewhere
      in this video (not invented out of common sense).

If family == "CR2" (temporal-ordering):
  C1. The question text must list a permutation A → B → C of THREE distinct
      events (each describable as a short phrase).
  C2. The gold-letter permutation must match the ACTUAL chronological order
      observed in support_chunks (cross-check timestamps).
  C3. The 3 events must be UNAMBIGUOUSLY identifiable in the chunks (not
      generic phrases like "person did something").

If family == "CR4" (compositional):
  C1. The question must require COMBINING ≥2 separate observations.
  C2. Each support_chunk must contribute a SUB-FACT needed to choose the
      correct option (no single chunk alone selects the gold letter).
  C3. Distractors must include single-chunk-correct options (e.g. "X only",
      "Y only") so the model is rewarded specifically for combining.

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "transient", "canonical_answer": "<letter>"}}
If ANY family-specific check fails, output: {{"valid": false}}"""

# CR3 (goal/intent) — answer is INFERRED from action sequence, not directly
# extractable from any single chunk. Standard verify rejects these as
# "not answerable from evidence". Use this prompt for CR3 only.
VERIFY_INTENT_PROMPT = """Verify this GOAL/INTENT question card (CR3).

Evidence chunks (early-to-mid video, the goal-revealing portion):
{evidence}

Card:
- question: "{question}"        # asks what the person is trying to accomplish
- canonical_answer: "{canonical_answer}"  (a single letter A/B/C/D)
- support_chunks: {support_chunks}

Goal/intent is INFERRED from the action sequence, not directly observable
in one chunk. Verify with these checks:

1. The question must explicitly ask about an OVERALL GOAL or activity
   ("What is being prepared?", "What sport is being trained?", "What is
   being assembled?") — NOT a single action ("What is the person doing
   right now?"; that's an E1 question, not CR3).

2. The option labeled canonical_answer must be the MOST PLAUSIBLE
   interpretation of the action sequence in the evidence.
   - For cooking: do the ingredients/tools shown match this dish?
   - For sports/DIY: do the actions shown match this activity?

3. The 3 distractor options must be FROM THE SAME GENRE/DOMAIN and share
   SOME early-stage actions with the correct answer (so the question
   actually requires integrating multiple observations, not just genre
   identification). E.g. for cooking: 4 different breakfast dishes that
   all involve eggs, not "guitar lesson" as a distractor.

4. canonical_answer is exactly one letter A/B/C/D.

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "persistent", "canonical_answer": "<letter>"}}
If the goal is too ambiguous, distractors are too far apart, or the
question is really an E1 rather than a goal question, output:
{{"valid": false}}"""

# N1 (HLD) verification — v9.4.1 OVO-aligned format.
# Card has 4 MC options: 3 plausible-but-absent + 1 "Unable to answer".
# Correct answer is the letter of "Unable to answer".
VERIFY_N1_PROMPT = """Verify this HALLUCINATION-DETECTION (OVO HLD) question card.

Evidence chunks (representative sample of the video):
{evidence}

Card:
- question: "{question}"
- canonical_answer: "{canonical_answer}"  (must be a single letter A/B/C/D)

Parse the four options A/B/C/D from the question text. Then check:
1. Does the question contain four inline options "A. ... B. ... C. ... D. ..."?
2. Is exactly ONE option literally "Unable to answer" (case-insensitive)?
3. Does canonical_answer point to that "Unable to answer" option?
4. Do the OTHER THREE options (the plausible-but-absent distractors) NOT
   appear in any chunk of the evidence above? If any distractor IS visible
   in the evidence, the card is INVALID — the question would have a real
   answer, defeating the "Unable to answer" gold.
5. Are the 3 distractors semantically plausible for this video genre? If
   they are wildly off-topic ("guitar" in a cooking video), the card is
   too easy — reject.
6. Is the question itself a normal content question (Where / What / Who /
   When), not a binary "Is X present?" form?

Output JSON only:
{{"valid": true, "support_chunks": [chunk_idx, ...], "visibility_type": "persistent", "canonical_answer": "<letter>"}}
If any check fails, output:
{{"valid": false}}"""


def _parse_verify_response(raw: Optional[str]) -> Optional[Dict]:
    """Parse verification response JSON."""
    if not raw:
        return None
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


async def _verify_one_card(
    card: Dict, evidence: List[Dict], client, video_id: str,
) -> Dict:
    """Verify one card independently. No cross-card dependency."""
    support = card.get("support_chunks", [])
    if not support:
        card["_verified"] = False
        return card

    family = card.get("family", "")
    is_negative = family in NEGATIVE_FAMILIES
    is_reasoning = family in {"CR1", "CR2", "CR4"}
    is_intent = family == "CR3"

    if is_negative:
        # N1: verify against the WHOLE video (or a wide sample), not just
        # support_chunks ± 2 — we need to confirm absence everywhere.
        all_idx = sorted(cap.get("chunk_idx", 0) for cap in evidence)
        # Sample up to 12 evenly-spaced chunks for context
        if len(all_idx) > 12:
            step = len(all_idx) // 12
            sampled = all_idx[::step][:12]
        else:
            sampled = all_idx
        ev_text = _format_evidence_for_prompt(evidence, sorted(sampled))
    elif family == "F6":
        # F6 (FPD): support_chunks is the SETUP; the continuation we need to
        # verify against is in chunks AFTER support_end. Extend the forward
        # range to support_end + 8 so the verifier can see both setup AND
        # actual continuation. (v9.3 — fixes batch1's 10/165 verify yield.)
        search_range = set()
        for sc in support:
            for c in range(max(0, sc - 2), sc + 9):
                search_range.add(c)
        ev_text = _format_evidence_for_prompt(evidence, sorted(search_range))
    elif is_intent:
        # CR3 (intent/goal): the goal is inferred from the FULL early-to-mid
        # action sequence, not from support_chunks alone. Pass up to 12
        # evenly-spaced chunks from the first 70% of the video.
        all_idx = sorted(cap.get("chunk_idx", 0) for cap in evidence)
        cutoff = max(1, int(len(all_idx) * 0.7))
        early = all_idx[:cutoff]
        if len(early) > 12:
            step = len(early) // 12
            sampled = early[::step][:12]
        else:
            sampled = early
        ev_text = _format_evidence_for_prompt(evidence, sorted(sampled))
    else:
        # Include support chunks ± 2 for context (works for CR1/CR2/CR4 too:
        # all chunks needed for reasoning are listed in support_chunks).
        search_range = set()
        for sc in support:
            for c in range(max(0, sc - 2), sc + 3):
                search_range.add(c)
        ev_text = _format_evidence_for_prompt(evidence, sorted(search_range))

    if not ev_text.strip():
        card["_verified"] = False
        return card

    if is_negative:
        prompt = VERIFY_N1_PROMPT.format(
            evidence=ev_text,
            question=card.get("question", ""),
            canonical_answer=card.get("canonical_answer", ""),
        )
    elif is_intent:
        prompt = VERIFY_INTENT_PROMPT.format(
            evidence=ev_text,
            question=card.get("question", ""),
            canonical_answer=card.get("canonical_answer", ""),
            support_chunks=card.get("support_chunks", []),
        )
    elif is_reasoning:
        prompt = VERIFY_REASONING_PROMPT.format(
            family=family,
            evidence=ev_text,
            question=card.get("question", ""),
            canonical_answer=card.get("canonical_answer", ""),
            support_chunks=card.get("support_chunks", []),
        )
    else:
        prompt = VERIFY_CARD_PROMPT.format(
            evidence=ev_text,
            question=card.get("question", ""),
            canonical_answer=card.get("canonical_answer", ""),
            answer_form=card.get("answer_form", "short_exact"),
            support_chunks=card.get("support_chunks", []),
            visibility_type=card.get("visibility_type", "transient"),
        )

    _verify_cfg = PASS_CONFIG.get("pass3a_verify", {})
    raw = await client._call_one(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_verify_cfg.get("max_tokens", 16384),
        temperature=_verify_cfg.get("temperature", 0.1),
        enable_thinking=_verify_cfg.get("thinking", False),
        request_id=f"{video_id}_verify_{card.get('card_id', '')}",
    )

    result = _parse_verify_response(raw)
    if not result or not result.get("valid", False):
        card["_verified"] = False
        return card

    # Apply fixes from verification
    num_chunks = max((cap.get("chunk_idx", 0) for cap in evidence), default=0) + 1
    fixed_sc = result.get("support_chunks")
    if isinstance(fixed_sc, list) and fixed_sc:
        # Bounds check: discard out-of-range chunk indices
        fixed_sc = [c for c in fixed_sc if isinstance(c, int) and 0 <= c < num_chunks]
        if not fixed_sc:
            logger.warning(f"  [{video_id}] verify {card.get('card_id')}: "
                           f"all support_chunks out of range, dropping card")
            card["_verified"] = False
            return card
        card["support_chunks"] = fixed_sc
    if result.get("visibility_type") in ("persistent", "transient"):
        card["visibility_type"] = result["visibility_type"]
    fixed_answer = result.get("canonical_answer")
    if isinstance(fixed_answer, str) and fixed_answer:
        card["canonical_answer"] = fixed_answer

    card["_verified"] = True
    return card


async def verify_cards(
    video_id: str, cards: List[Dict], evidence: List[Dict], client,
) -> List[Dict]:
    """Verify all cards concurrently. Each card is an independent 397B call.

    Drops invalid cards, fixes support_chunks/visibility_type/canonical_answer.
    Returns only verified cards.
    """
    if not cards:
        return []

    tasks = [_verify_one_card(card, evidence, client, video_id)
             for card in cards]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verified = []
    dropped = 0
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"  [{video_id}] verify card failed: {result}")
            dropped += 1
            continue
        if result.get("_verified", False):
            verified.append(result)
        else:
            dropped += 1

    logger.info(f"  [{video_id}] 3-A verify: {len(verified)} passed, {dropped} dropped")
    return verified
