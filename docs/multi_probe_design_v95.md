# v9.5 — Multi-Probe Family Design (OVO REC / CRR / SSR + OJR holding)

> **STATUS — DEPRECATED (Historical only)**
>
> This document was the original v9.5 design that proposed extending F5 / E2 /
> P1 with optional `probes` / `step_probes` schema fields. The implementation
> took a different path:
>
> - **F5 (REC)** — kept as a single-question family, added `multi_response`
>   dispatch with `progressive_answers` map (per-probe gold = cumulative count
>   computed in pass3b from `support_chunks`). No schema change.
> - **SSR alignment** — implemented as a NEW family **F7** (binary "Has step X
>   been done by now?" + `step_chunk` extra field), not as extension of P1.
> - **CRR alignment** — implemented as a NEW family **CR5** (descriptive
>   clue-delayed multi-probe + `ask_chunk` / `clue_chunk` extras), not as
>   extension of E2.
> - **OJR holding** — addressed via the v9.5 fine-grained `desc` requirement
>   in the pass1a EVIDENCE_GRAPH_PROMPT (color + material + pattern + style +
>   condition); F2 prompt change wasn't necessary.
>
> See **`docs/design.md`** for the current design. This file kept for
> historical context; do **not** consult for live behaviour.

## 1. Why this exists (historical context)

After v9.4, our 18 families cover **9 of 12 OVO tasks** structurally. Three
remaining tasks require a structural extension to the card schema, not just
another prompt:

| OVO task | n samples | probes / sample | shape |
|---|---|---|---|
| REC | 82 | avg 8.5, max 33 | cumulative integer count, same question |
| SSR | 42 | avg 15 | per-step Yes/No, **question varies per probe** |
| CRR | 48 | avg 5 | binary Y/N, gold flips at `clue_time`, same question |

A v9.4 card carries one `(question, canonical_answer, support_chunks)`
tuple. OVO above wants `N (realtime, expected_value)` tuples per card, so
the model is trained to maintain its answer over time as new evidence
arrives. That is **fundamentally what streaming agents are for** — and
right now we're not training for it.

Plus 1 prompt-level gap: **OJR holding** (~44/184) — "What is X holding /
wearing?" — slot under existing F2 but the prompt doesn't currently
solicit possession-state questions.

Goal: by v9.5, our generated data covers **all 12 OVO tasks** at the
distribution level, not just 9.

## 2. Scope

| # | Family | New / extend | Schema impact | OVO target |
|---|---|---|---|---|
| F5b | extend F5 with `probes` | additive (optional field) | REC |
| E2b | extend E2 with `probes` | additive | CRR |
| P1b | extend P1 with `step_probes` | additive (different per-probe question) | SSR |
| F2 | prompt enrichment only | none | OJR holding |

No new family CODES (we extend the existing ones with optional fields), so
existing dispatch / RETENTION_CLASS / FAMILY_TO_OVO entries stay valid.

## 3. Card schema (additive — old cards still load)

Today (v9.4):

```jsonc
{
  "card_id": "video_F5_001",
  "family": "F5",
  "question": "How many times has the chef stirred?",
  "canonical_answer": "3",
  "answer_form": "number",
  "support_chunks": [12, 24, 38],
  "visibility_type": "transient"
}
```

v9.5 (new optional `probes` / `step_probes`):

```jsonc
// REC (F5 with probes)
{
  "card_id": "video_F5_001",
  "family": "F5",
  "question": "How many times has the chef stirred so far?",
  "answer_form": "number",
  "support_chunks": [12, 24, 38],
  "visibility_type": "transient",
  "probes": [                         // NEW
    {"chunk_idx": 14, "expected": "1"},   // after 1st occurrence
    {"chunk_idx": 26, "expected": "2"},   // after 2nd
    {"chunk_idx": 40, "expected": "3"},   // after 3rd
    {"chunk_idx": 50, "expected": "3"}    // late: count stable
  ]
  // canonical_answer is now redundant for multi-probe — derived from
  // probes[-1].expected for backwards-compat readers (e.g. pass3d IFD).
}

// CRR (E2 with probes)
{
  "card_id": "video_E2_001",
  "family": "E2",
  "question": "Does the woman walk past the car?",
  "answer_form": "binary",
  "support_chunks": [148, 154],         // ask_chunk, clue_chunk
  "visibility_type": "transient",
  "ask_chunk": 148,                     // realtime/2 of ask_time
  "clue_chunk": 154,                    // realtime/2 of clue_time
  "probes": [                           // NEW
    {"chunk_idx": 148, "type": 0, "expected": "No"},   // before clue
    {"chunk_idx": 151, "type": 0, "expected": "No"},
    {"chunk_idx": 155, "type": 1, "expected": "Yes"},  // after clue
    {"chunk_idx": 159, "type": 1, "expected": "Yes"},
    {"chunk_idx": 169, "type": 1, "expected": "Yes"}
  ]
}

// SSR (P1 with step_probes — question varies per probe)
{
  "card_id": "video_P1_001",
  "family": "P1",
  "answer_form": "binary",
  "tutorial": "PutOnHairExtensions",
  "all_steps": [
    "pull up the hair",
    "put on extensions",
    "comb down"
  ],
  "step_intervals": [
    {"step": "pull up the hair",  "start_chunk": 12, "end_chunk": 15},
    {"step": "put on extensions", "start_chunk": 15, "end_chunk": 24},
    {"step": "comb down",         "start_chunk": 53, "end_chunk": 58}
  ],
  "support_chunks": [12, 15, 53],
  "visibility_type": "transient",
  "step_probes": [                       // NEW (per-probe question)
    {"chunk_idx": 13, "step": "pull up the hair", "expected": "Yes"},
    {"chunk_idx": 15, "step": "pull up the hair", "expected": "Yes"},
    {"chunk_idx": 13, "step": "put on extensions", "expected": "No"},
    {"chunk_idx": 17, "step": "put on extensions", "expected": "Yes"},
    {"chunk_idx": 60, "step": "comb down",         "expected": "No"}
    // ...
  ]
  // No top-level question — built per-probe via a fixed template
  // (see SSR_QUESTION_TMPL below).
}
```

**Schema rules**:
- Multi-probe families have ≥2 entries in `probes` / `step_probes`.
- `chunk_idx` is the per-probe ask point. All probe chunk_idx must be
  ≤ `num_chunks - 1` (validated in pass3a).
- `expected` follows the family's `answer_form` (number / binary).
- For SSR (`step_probes`), `step` carries the per-probe question variable;
  the question text is materialised at sample-render time.
- For E2/CRR, `ask_chunk` / `clue_chunk` are denormalised so pass3b
  doesn't need to scan probes.

## 4. Per-family probe semantics

### 4.1 F5 / REC — cumulative count

**Why probes**: an OVO REC sample evaluates the same question at multiple
times; gold count is monotonically non-decreasing, increments at the
moment the action's nth occurrence completes.

**Probe-time selection** in pass3a `classify_chunks` for REC:
1. Detect ≥2 occurrences of the same primary-action lemma (`_action_verb_lemma`).
2. Each occurrence interval `[start_i, end_i]` becomes a "checkpoint":
   add a probe at `chunk_idx = end_i + 1` with `expected = i + 1`.
3. Add 1–2 "stable" probes far past the last occurrence (count stays the
   same) to teach "no double-counting".

**Generation**: still a single 397B call to FAMILY_PROMPTS["F5"]; the
prompt now also asks for `probes` field. Teacher LLM places probes at
post-occurrence chunks.

### 4.2 E2 / CRR — clue-reveal Y/N

**Why probes**: the question is asked at `ask_time`, but the answer is
only available after `clue_time`. Before clue → "No" (or honest refusal);
after clue → "Yes".

**Probe-time selection** in pass3b for CRR:
- 2 probes at `[ask_chunk, ask_chunk+2]` with `type=0`, `expected="No"`
- 2–3 probes at `[clue_chunk+1, clue_chunk+5, clue_chunk+15]` with
  `type=1`, `expected="Yes"`

**Why this is harder than v9.4 E2**: existing E2 puts the model in
`event_watch` mode — silent silent silent → response at trigger. For CRR
multi-probe, the model needs to actively *answer "No"* before the clue,
not stay silent. That's a different action distribution.

### 4.3 P1 / SSR — per-step Y/N

**Why probes**: each probe asks about a *different* step. Question text
differs per probe. The shared context is the tutorial + the visible
action timeline.

**Question template** (materialised at render time):

```
SSR_QUESTION_TMPL = (
  "You're watching a tutorial which contains a sequence of steps. "
  "The following is one step from the procedure:\n\n"
  "{step}\n\n"
  "Is the person currently carrying out this step? Yes/No."
)
```

**Probe-time selection** in pass3a for SSR:
For each step `s_i` with interval `[start_i, end_i]`:
- 1–2 probes inside interval (`expected="Yes"`)
- 1–2 probes outside interval (`expected="No"`) — pick from
  - mid-other-step (e.g. during step `j ≠ i`)
  - before any step starts
  - after step `i` ends

Cap total probes per card at ~12 to keep trajectory length sane.

### 4.4 F2 / OJR holding — prompt enrichment only

In `FAMILY_PROMPTS["F2"]`, add to the procedure list:

```
Genres of attribute questions to mix into the {n} cards:
- color / material / size (existing)
- POSSESSION-STATE: "What is the person holding / wearing / using
  right now?" — answer is the object itself, drawn from
  visible_entities at support_chunks. Distractors are other entities
  visible at OTHER chunks.
```

No schema change. ~5-line prompt edit + an example. This handles ~44/184
OJR samples.

## 5. Pipeline changes

### 5.1 pass3a `pass3a_cards.py`

- **Card prompts**: F5 / E2 / P1 prompts updated to also emit `probes`
  array (with 2–10 entries). Output schema gains a single optional field.
- **classify_chunks**: REC and SSR get richer scans:
  - REC: detect repeat occurrences with end-chunks → seed candidate probes
  - SSR: detect step intervals (already existing P1 fallback path) →
    seed step_probes
- **Card validators in `_generate_family_cards`**: enforce probes field
  shape when family supports it; if probes missing in a multi-probe card,
  fall back to single-probe (build `probes = [{chunk_idx: derived_ask,
  expected: canonical_answer}]`) and log a warning.
- **VERIFY prompts**: F5/E2/P1 verify prompts gain a "verify probes
  list" check — each probe's `expected` must be derivable from evidence
  up to that probe's chunk.

Cache version: 3a → **v9.5**. All downstream stages bump too.

### 5.2 pass3b `pass3b_placement.py`

The single-probe placement logic stays for v9.4 cards. New logic for
multi-probe:

```python
def _multi_probe_placement(card, rollout, evidence) -> Dict:
    """Multi-probe cards expand into ONE trajectory with one ask + N
    response chunks. The ask happens at the FIRST probe (or ask_chunk
    for E2 CRR). All subsequent probes are response samples in the
    same trajectory sharing the same card_id."""
    probes = card.get("probes") or card.get("step_probes")
    if not probes:
        return None
    # ask = first probe's chunk_idx (or card.ask_chunk for E2)
    ask_chunk = card.get("ask_chunk") or probes[0]["chunk_idx"]
    return {
        "card_id": card["card_id"],
        "ask_chunk": ask_chunk,
        "sequence_type": "multi_probe",   # new sequence_type
        "key_chunks": {
            "ask": ask_chunk,
            "probe_chunks": [p["chunk_idx"] for p in probes],
            "post_silent": min(probes[-1]["chunk_idx"] + 1, num_chunks - 1),
        },
        "family": card["family"],
        "support_chunks": card["support_chunks"],
        "availability": "in_visual",   # first probe is at-time
        "difficulty_tier": "multi_probe",   # new tier label for stats
    }
```

**Trajectory budgeting** (`plan_trajectories`): a multi-probe card
contributes max 1 trajectory by itself (already a long traj of N samples).

### 5.3 pass3c `pass3c_samples.py`

New `sequence_type == "multi_probe"` handler in
`generate_trajectory_samples`. Skeleton:

```python
elif seq == "multi_probe":
    probes = card.get("probes") or card.get("step_probes") or []
    # First probe: emit a "response" sample with the question injected.
    # For SSR, build per-probe question via SSR_QUESTION_TMPL.
    # For REC/CRR, the question is the same across probes.
    for i, probe in enumerate(probes):
        c = probe["chunk_idx"]
        if "step" in probe:
            q = SSR_QUESTION_TMPL.format(step=probe["step"])
        else:
            q = card["question"]
        # Determine response text via _normalize_exact_form_answer
        # (probe.expected is already the canonical letter / Yes-No / digit).
        resp = _normalize_exact_form_answer(probe["expected"],
                                              card["answer_form"])
        # Emit sample at chunk c
        samples.append(_make_sample(
            c, "SYSTEM_PROMPT", "response", base_think, queries_state,
            response=resp,
            user_input=q if i == 0 or "step" in probe else None,
            ...
        ))
        if i == 0:
            _add_query(q, c, answer=resp, response_chunk=c)
        else:
            queries_state[-1]["answers"].append({"text": str(resp), ...})
        # Insert silent base samples between probes if gap > 4 chunks
        ...
```

**Shared `<recall>` between probes**: REC trajectories may need recall
when probe_i depends on remembering occurrence at chunk_j far in the past.
Reuse existing `recall_success` / `recall_fail_then_found` plumbing
*within* a multi_probe trajectory by inserting recall sub-sequences
between probes. Defer this to v9.5.1 — start with no recall in
multi_probe.

### 5.4 pass4 `pass4_verify.py`

Add per-probe checks:
- response strict format check (already exists for per-sample)
- multi-probe gold consistency:
  - REC: probes' `expected` is monotonically non-decreasing across
    `chunk_idx`.
  - CRR: probes with `type=0` have `expected="No"`, `type=1` have
    `expected="Yes"`; the type flips exactly once around `clue_chunk`.
  - SSR: probe's `expected="Yes"` ⇒ `chunk_idx` ∈ `[start_chunk, end_chunk]`
    of the matching step.

These are pure-program checks (no LLM).

### 5.5 render_samples + stream_data_processor

The render side already handles per-sample `gold_answer`. For multi-probe,
each probe is an independent sample with its own `gold_answer`, so render
needs no change. The metadata gets a new field:

```jsonc
"metadata": {
  "answer_form": "number",
  "family": "F5",
  "probe_index": 2,               // NEW: which probe (0-based) this sample is
  "n_probes": 4,                  // NEW
  ...
}
```

### 5.6 pass3d / IFD

Multi-probe samples scale-multiply: a card with 8 probes becomes 8
trainable samples. To prevent any one card from dominating IFD selection
(diversity loss), add a per-card cap in `pass3d_select.select_samples`:
"keep at most 4 samples per card_id by default". Existing diversity logic
already has card_id awareness — extend to multi-probe.

## 6. Eval integration

`scripts/eval/ovo/eval_full.py` already has the right per-task
evaluators (`eval_rec`, `eval_ssr`, `eval_crr`) that walk the agent across
the right probe times and score per-probe. So **no eval-side changes**;
the v9.5 SFT data brings TRAINING in line with how eval already works.

`scripts/eval/test_set_agent.py` (our test.jsonl walker) handles
single-probe samples. For multi-probe samples, score the FIRST probe's
gold (since test.jsonl entries are still per-sample). This is a soft
limitation: if you want to evaluate the multi-probe training signal on
test.jsonl, you'd render ONE sample per probe and score per-sample.
Already the case after pass3c expands probes.

## 7. Migration path

v9.4 cards (no probes field) and v9.5 cards (with probes) coexist:

- pass3b: `if "probes" in card or "step_probes" in card: _multi_probe_placement(...) else: existing()`
- pass3c: `if seq == "multi_probe": new_handler() else: existing()`
- pass4: `if metadata.n_probes: multi-probe consistency check else: skip`

**Re-running**: bump 3a→v9.5 invalidates cards. Re-run from 3a:

```bash
python -m scripts.agent_data_v5.pipeline run --force_rerun_from 3a
```

Pass1a/1b/2 untouched. Cost ~ same as v9.4 batch1 rerun.

## 8. Tests

Add to `tests/test_v95_changes.py`:

1. **Card schema validation**: F5/E2/P1 cards accept `probes` field;
   probes monotonicity for REC; probe-type/expected coupling for CRR.
2. **Multi-probe placement**: pass3b emits `sequence_type=multi_probe` with
   `key_chunks.probe_chunks` listing all probe chunk_idx.
3. **Sample generation count**: a 5-probe card → 5 response samples
   (plus base silents).
4. **Per-probe gold**: each emitted sample has the right
   `metadata.gold_answer` matching its probe's expected.
5. **SSR question template**: per-probe question varies as expected;
   shared trajectory still has consistent system prompt.
6. **Pass4 monotonicity**: REC card with non-monotonic expected list
   gets rejected; CRR card with mismatched type/expected gets rejected.
7. **OJR holding F2 prompt**: prompt mentions "holding" / "wearing".

## 9. Risks / open questions

1. **Trajectory length explosion**. SSR avg 15 probes/card × 42 sample
   videos = 630 samples just for SSR. With base silent fillers between
   probes, that's ~5000 samples/SSR-video. Need a probe cap and silent
   subsampling. Default: `max_probes_per_card = 8`.

2. **Recall in multi-probe**. REC's late probes legitimately need
   recall (count an action that happened 60s ago). v9.5.1 adds recall
   sub-sequences inside multi_probe trajectories. v9.5 core: no recall —
   probes too close to the event to need it.

3. **Compress in multi-probe**. A 30-probe REC trajectory will trigger
   compression mid-way. The trigger format matches v9.4 (system-injected
   `<compress_trigger>`). Already compatible.

4. **GRPO reward for multi-probe**. RL eval gives a reward per response
   sample. A multi-probe rollout produces N response samples with
   per-sample correctness rewards. Existing `silent_quality` handles
   "should respond" correctly. Existing `correctness` handles per-letter
   match. **No reward changes needed**.

5. **Teacher reliability for probe placement**. Asking the teacher LLM
   to place 5+ probes correctly inside a generation prompt is harder
   than asking for one canonical_answer. Mitigation: pass3a shifts the
   probe-time selection to *deterministic Python* (steps 4.1/4.2/4.3),
   leaving the teacher only for question text + step descriptions.

## 10. Implementation phases

Phase A — schema + structural (no LLM-prompt changes):
- Card schema accepts `probes` / `step_probes` (additive)
- pass3b emits multi_probe placement
- pass3c emits N samples per probe
- pass4 multi-probe consistency check
- v9.5 cache bump
- Tests 1, 2, 3, 4, 6

Phase B — population (LLM prompts):
- F5 prompt asks for probes (REC cumulative)
- E2 prompt asks for probes (CRR ask/clue)
- P1 prompt asks for step_probes (SSR per-step)
- Tests 5

Phase C — F2 OJR holding:
- F2 prompt enrichment
- Tests 7

Phase D — recall inside multi-probe (deferred to v9.5.1).

Phase A is ~400 LoC of structural code, B is prompt edits + classify_chunks
shims, C is 5-line edit. Total: ~1 day implementation + dry-run.

## 11. Acceptance

After v9.5 batch rerun, OVO eval should show:

- HLD: now ≥30% (was ~0% with v9.4 inverted prompt)
- REC: per-probe accuracy meaningful number (was untrained → near
  random on cumulative count)
- SSR: per-probe Y/N accuracy meaningful (was untrained)
- CRR: type=0 / type=1 separation reported (FP rate < 50% required for
  the model to actually be using the clue, not just defaulting Yes)

If any of those four still tanks after v9.5, we've got a deeper signal
problem (insufficient probe density, reward mis-tuning, or eval-vs-train
mismatch we missed) and we should triage before another iteration.
