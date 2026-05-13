"""Cache version fingerprinting (v9.1).

Each pass writes a `_version` file in its output directory. On load, if
the marker doesn't match the current version, the cached payload for
that video is invalidated (returns None) so the pipeline regenerates.

Granular versions: bump only the stage(s) you changed. Downstream caches
are invalidated automatically because they share parent dirs.

Usage in save_X():
    save_1a(video_id, evidence)
    write_stage_version("1a")   # idempotent

Usage in load_X():
    if not stage_version_ok("1a"):
        return None  # cache stale
    ...read file...
"""

from pathlib import Path
from typing import Dict

from .config import (
    DATA_ROOT,
    EVIDENCE_1A_DIR,
    EVIDENCE_1B_DIR,
    ROLLOUT_DIR,
    TASK_CARDS_DIR,
    PLACEMENTS_DIR,
    SAMPLES_3C_DIR,
    VERIFIED_DIR,
    FINAL_DIR,
)

# ---------------------------------------------------------------------------
# Per-stage version. Bump when you change the stage's prompt/code that
# would invalidate prior outputs. Downstream stages auto-invalidate.
# ---------------------------------------------------------------------------

STAGE_VERSIONS: Dict[str, str] = {
    # v12.11 audit-5 P1 #5 (2026-05-01): bumps below align with the v12.11
    # data-construction logic changes. Without these, an existing cluster
    # cache stamped v12.5 would silently reuse stale outputs:
    #   v12.43 (2026-05-06): prompt rendering splits old <queries> into
    #        <active_query> plus <response_history>. Only the live query is
    #        rendered; closed historical Q&A disappears after the final answer.
    #        Regenerate all *_messages.jsonl and RL parquets that freeze prompts.
    #   v12.42 (2026-05-06): system prompts spell out exact ordinary answer,
    #        silent answer, recall tool JSON, and compression tool JSON
    #        grammars. Regenerate all *_messages.jsonl and RL parquets that
    #        freeze prompts.
    #   v12.41 (2026-05-06): pass5/rendered messages use separate ordinary
    #        streaming and compression-only system prompts. Compression turns
    #        keep only a bare <compress_trigger/> in user_input while the
    #        compression rules move to the system prompt. Regenerate all
    #        *_messages.jsonl and RL parquets that freeze prompts.
    #   v12.44 (2026-05-07): pass3 redesign updates card generation,
    #        placement/sample construction, and verification-facing metadata.
    #        Regenerate pass3a and all downstream artifacts from existing
    #        pass1/pass2 caches.
    #   v12.45 (2026-05-07): pass3c hardens selected recall slots against the
    #        exact pass2 memory snapshot seen at the answer chunk. Easy recall
    #        cards are replaced in-place with support-grounded questions whose
    #        answers are absent from current memory, while preserving ask/answer
    #        chunks, family, mechanism, and trajectory counts. Regenerate 3c
    #        and downstream render/verification outputs from existing 3a/3b.
    #   v12.46 (2026-05-07): pass3c no longer drops a whole trajectory when an
    #        easy recall slot cannot be hardened. It preserves the fixed
    #        ask/answer chunks and downgrades that slot to memory_direct, while
    #        pipeline fails fast on any remaining 3c trajectory/sample loss.
    #   v12.49 (2026-05-07): pass3b reserves non-recall HLD/abstention slots
    #        with deterministic sampling, keeping HLD near the previous
    #        reasonable family share without counting it as recall.
    #   v12.50 (2026-05-07): pass5/render and runtime prompt builders make
    #        inter-chunk compression text-only: no visual_window, images,
    #        videos, active query, or recalled-frame context. Regenerate all
    #        *_messages.jsonl and RL parquets that freeze prompts.
    #   v12.51 (2026-05-07): pass5 splits multi-turn recall SFT into
    #        recall_query rows with recall tools and post_recall rows with
    #        no tools plus last-assistant-only loss, matching runtime
    #        post-recall turns.
    #   v12.52 (2026-05-07): MC option rebalancing also rewrites
    #        per_emit_answers values in flat metadata and trajectory questions,
    #        keeping chunk-level RL gold answers aligned with the
    #        rebalanced correct_option/options and rendered <answer>.
    #   v12.53 (2026-05-07): pass3c recall hardening asks for multiple
    #        memory-novel replacement candidates per selected recall slot,
    #        ranks historical evidence by current-memory novelty, and repairs
    #        answer-leaking recall queries before falling back to memory_direct.
    #   v12.54 (2026-05-07): pass3c validates every selected recall slot,
    #        including already-hard cached cards, against memory plus the
    #        current visual think before the recall tool call. This blocks
    #        recall samples whose answer is already visible in the current
    #        frame/context and catches short/non-ASCII answer leaks.
    #   v12.55 (2026-05-07): streaming prompt explicitly prioritizes recall
    #        when current visible evidence is insufficient, compression prompt
    #        states that compress is mandatory on compression turns, and
    #        pass3c/pass5 render recall_query first-turn thinks as
    #        current-frame text memory plus action-aware recall decisions
    #        instead of question-blind visual captions alone. Regenerate
    #        pass3c or at least pass5 rendered messages.
    #        The recall-result second turn also uses a separate
    #        recall_response system prompt with no tool action space.
    #   v12.56 (2026-05-07): pass5 uses the clearer post_recall metadata
    #        alias for the no-tools turn after recall.
    #   v12.57 (2026-05-07): pass5 marks recall_query and post_recall with
    #        separate loss_class metadata. SFT uses loss_class for class
    #        weighting/diagnostics and masks assistant spans through <|im_end|>
    #        without also training the following newline.
    #   v12.58 (2026-05-07): pass5/runtime prompts add explicit mode headers
    #        for streaming QA, post-recall decision, and memory-maintenance
    #        compression turns. Regenerate pass5 rendered messages so SFT/RL/eval
    #        see the stronger mode separation text.
    #   v12.59 (2026-05-08): pass5/runtime prompts soften recall/compress
    #        exploration wording after rollout diagnostics: recall is encouraged
    #        when historical evidence could help, and compress range selection
    #        asks for the contiguous range whose replacement least hurts later
    #        reasoning rather than hard-coding rollout-specific age windows.
    #        Regenerate pass5 rendered messages and RL/eval parquet prompts.
    #   v12.60 (2026-05-10): pass3a teacher prompts use full pass1/pass1b
    #        timeline context and stricter MC/question-form constraints;
    #        pass3b schedules/filters only memory-novel recall slots while
    #        preserving card gold answer chunks; pass3c renders recall-silent
    #        reasons; pass5 balances pending/post-answer/no-question silence
    #        and keeps open multi-emit intervals as pending.
    #   v12.61 (2026-05-10): pass3B trajectory selection adds a hard minimum
    #        ask gap, strengthens temporal spread scoring, and lets very long
    #        videos select up to 20 questions without densifying short videos.
    #   v12.62 (2026-05-10): pass3a prompts push OVO-heavy STU/OJR/OCR-style
    #        cards toward event-anchored fine visual details that survive later
    #        recall placement; pass3B boosts OVO-heavy family selection and
    #        keeps hard memory-direct recall probes behind an opt-in filter
    #        experiment for dirty-memory studies.
    #   v12.63 (2026-05-10): pass3a asks OVO-heavy OJR/STU/ACR families for
    #        two diverse candidates per video and explicitly pairs current-style
    #        cards with event-anchored historical-detail cards when possible.
    #   v12.64 (2026-05-10): pass3a prompts add OVO weak-case guidance for
    #        HLD, OCR, CR1/CR2/CR4/CR5, F7, and CRR1. pass3b preserves
    #        existing F7/CRR1 far-after Yes probes when available instead of
    #        raising status-card sampling rates.
    #   v12.65 (2026-05-10): pass3a adds benchmark-like but not
    #        benchmark-specific question/style guidance: compact user wording,
    #        varied temporal anchors, hard same-type MC options, and a small
    #        exploratory mix of multi-clue / before-after / insufficient-evidence
    #        questions.
    #   v12.66 (2026-05-10): pass3a evidence prompts compact over the full
    #        video instead of prefix-truncating long timelines; pass3c recall
    #        hardening uses the same compact question/option style constraints
    #        and parallelizes selected hardening calls per trajectory.
    #   v12.67 (2026-05-10): pass3 context estimates are updated from a
    #        batch1-8 prompt audit, and pass3c teacher output caps are
    #        right-sized for response, recall-query, and recall-hardening calls.
    #   v12.68 (2026-05-10): pass3b preserves HLD/Unable recall evidence
    #        checks instead of dropping them on BM25/memory-overlap filters;
    #        pass3c rejects answer-leaking recall_query payloads; MC rebalance
    #        validates every answer/options rewrite before final files are
    #        accepted.
    #   v12.69 (2026-05-10): pass3e requires returned historical chunks for
    #        every recall sample; pass5 hard-fails if
    #        active_query/options/answer-format are missing or rendered more
    #        than once in final messages.
    #   v12.70 (2026-05-10): compression samples remove raw visual/query/recall
    #        payloads from input so SFT, RL, and eval see the same text-only
    #        memory-maintenance boundary. Ordinary visual turns intentionally
    #        keep text memory even when recent/current visual chunks overlap.
    #   v12.71 (2026-05-10): recall_result becomes metadata-only in rendered
    #        samples/messages; recalled_frames carry the visual evidence.
    #   v12.72 (2026-05-10): pass3 raises non-HLD visual-verification recall
    #        probes, promotes kept memory-direct visual checks into recall_demo,
    #        preserves multi-event history recall despite text-memory overlap,
    #        and pass5 lowers SFT patrol-silent sampling so recall/compress rows
    #        are not drowned by ordinary silent chunks.
    #   v12.73 (2026-05-10): pass3 adds a soft non-MCQ trajectory floor focused
    #        on active-responding REC/SSR/CRR-style F5/F7/CRR1 questions;
    #        final video splits are profile-balanced across SFT/RL/val/test;
    #        SFT silent downsampling preserves boundary subtypes/families and
    #        never drops recall/compress rows.
    #   v12.48 (2026-05-07): pass3b reserves one non-recall HLD/abstention
    #        slot when available, so HLD keeps a reasonable family share
    #        without being counted as successful recall supervision.
    #   v12.47 (2026-05-07): HLD1 / Unable-to-answer cards are excluded from
    #        successful recall_demo supervision. pass3b turns their historical
    #        variants into memory_direct before trajectory selection, and
    #        pass3c applies the same downgrade for stale 3b caches.
    #   v12.40 (2026-05-06): all non-HLD multiple-choice cards now place the
    #        correct option into a stable A/B/C/D slot and sort distractors by
    #        stable hash before validation/rendering. This removes LLM/heuristic
    #        answer-letter bias for C1/N1/P1/CR*/R1/ACR/STU/OJR while preserving
    #        the correct answer text.
    #   v12.39 (2026-05-06): HLD1 normalizes the Unable-to-answer option into
    #        a stable A/B/C/D slot by card_id before validation/rendering. This
    #        removes LLM letter-position bias while preserving diverse LLM
    #        question wording and concrete distractors.
    #   v12.38 (2026-05-06): HLD1 returns to LLM generation with a stricter
    #        OVO-HLD prompt that asks for diverse unanswerable location,
    #        placement/object, state, count, color/attribute, and before-memory
    #        MC negatives. The deterministic HLD factory is now only a fallback
    #        after schema/evidence verification, so pass3a+downstream caches
    #        must be rebuilt to pick up the new question distribution.
    #   v12.37 (2026-05-06): pass3a rejects HLD1 cards when any concrete
    #        non-Unable option is already supported by the grounding evidence,
    #        asks for up to two C1/OCR cards per video, and raises F7/SSR
    #        adoption/selection while lowering HLD selection pressure. This
    #        requires regenerating cards and downstream placements/samples.
    #   v12.36 (2026-05-06): pass3b no longer injects recall+silent wait
    #        turns for silent_then_response/event_watch questions, because
    #        their answer support is in the future and recall cannot be the
    #        minimal action. pass3e also relaxes compression summary verifier
    #        heuristics so concise compress tool supervision is not rejected
    #        for style words, time indices, or short policy-think text.
    #   v12.35 (2026-05-06): pass2 visible-memory rollout now recompresses
    #        over the unified timeline, persists source_chunks/merge_level,
    #        and pass3e accepts not_yet recall waits that return historical
    #        frames. This changes rollout, compression SFT, recall rendering,
    #        verification, and final pass5 artifacts, so stages 2-5 must
    #        rerun from the existing pass1 evidence.
    #   v12.34 (2026-05-06): pass3c repairs malformed/stale recall-response
    #        query/result payloads into legal historical recall samples using
    #        grounding_frames/support_chunks/gold emits. Unrecoverable recall
    #        samples now raise instead of silently becoming plain SFT responses.
    #   v12.33 (2026-05-06): pass3b treats direct answers as valid only while
    #        support remains inside the visual window. Backward/memory mid-band
    #        questions now use recall instead of answering directly from recent
    #        thinks, aligning recall-response with historical visual evidence.
    #   v12.32 (2026-05-06): pass3b validates placement timing after answer
    #        chunk normalization. Recall placements require support to be in
    #        the past and outside the visual window; direct/forward placements
    #        must respect their ask/support/response ordering.
    #   v12.31 (2026-05-06): MC canonical_answer is normalized to the exact
    #        correct option text, and single_emit answer chunks are normalized
    #        to the latest grounding frame so recall-response support is
    #        guaranteed to be historical when recall fires.
    #   v12.30 (2026-05-06): pass3c recall tool calls are hard-bounded to
    #        past evidence only. LLM recall_query outputs are rejected if
    #        their time_range ends after the current chunk, and noisy recall
    #        distractors cannot be sampled from future chunks. Rendered
    #        metadata now falls back from support_chunks to grounding_frames,
    #        and pass3e verifies recall_result time/returned_chunks directly.
    #        Legacy filter_samples is now a tag-only compatibility alias so
    #        no verification path drops rows and creates trajectory gaps.
    #        Invalid recall-response queries/results were guarded, and
    #        malformed compression ranges are rebuilt from compressed_thinks_chunks.
    #   v12.29 (2026-05-06): pass3 card taxonomy is rebalanced around
    #        benchmark-compatible families: HLD1 unanswerable MC negatives,
    #        F7 SSR-style No->Yes multi-time status, ACR/STU/OJR direct MC
    #        families, and C1 MC OCR. pass3a now rejects malformed family
    #        outputs and falls back per family; pass3b boosts rare SSR/HLD/OCR
    #        selection.
    #   v12.28 (2026-05-05): recall_silent is restored only as a non-terminal
    #        wait state: a forward question may call recall, receive not_yet,
    #        keep the query open, and answer at a later grounded chunk.
    #   v12.27 (2026-05-05): pass3 requires every card/placement/question
    #        to have grounded answer emits. Production recall demos now sample
    #        only oracle/noisy retrieval results; recall-failure/no-answer
    #        trajectories are rejected instead of entering SFT/RL/eval.
    #   v12.26 (2026-05-05): pass5/render + SFT/RL/eval support late-bound
    #        AB visual carriers for the same pre-extracted frames:
    #        timestamped image list (`ts_image`) and native Qwen video block
    #        with explicit video_metadata (`video_meta`). Pending queries now
    #        render Answer format instructions consistently across SFT, RL,
    #        and eval/test prompts.
    #   v12.25 (2026-05-05): pass1a emits a current-only `think`
    #        observation-note JSON field per chunk. This is supervised text,
    #        not Qwen/vLLM enable_thinking reasoning. pass2 consumes that
    #        pass1 note directly and only performs timeline/token accounting
    #        plus text-only compression summary generation. This removes the
    #        unstable pass2 teacher observation call that mixed full text
    #        memory with the visual window and caused stale-repeat thinks.
    #   v12.24 (2026-05-05): pass2 teacher still keeps full compression,
    #        full memory, and the same 16s sliding visual window, but the
    #        OBSERVATION / REPAIR prompts now see memory as a structured
    #        archival ledger instead of continuation-friendly prose. This is
    #        a teacher-only anti-copying change; student/shared protocol stays
    #        unchanged until we verify the repetition reduction.
    #   v12.22 (2026-05-04): all construction/rendering stages align on the
    #        project-wide pre-extracted-frame protocol: frame-tag text before
    #        each image/image_url. pass1a/pass2 teacher calls, pass5 SFT
    #        messages, SFT/RL/eval/deploy renderers now share the same helper
    #        instead of mixing Qwen video blocks with image lists. This avoids
    #        vLLM pre-sampled-video metadata drift while preserving real 2fps
    #        temporal anchors for the model.
    #   v12.21: pass3 trajectory planning enforces one
    #        active question at a time, prevents answer-chunk collisions,
    #        preserves open multi-answer query status, raises recall tool-use
    #        coverage, fixes F7/progress binary verification, and consumes the
    #        v12.18 pass2 timestamped-image rollout snapshots downstream.
    #   v12.18 background: pass3 display taxonomy fields, mixed MC answer
    #        protocols (letter/text/letter+text), semantic gold_answer split
    #        from SFT target, and verifier/rebalance updates.
    "1a": "v12.25",
    "1b": "v12.25",
    "2":  "v12.35",
    "3a": "v12.73",
    "3b": "v12.73",
    "3c": "v12.73",
    "4":  "v12.73",  # canonical key — verification/final split render
    "5":  "v12.74",  # pass5 multi-turn trajectory render version
}
# v12.11 review-fix (2026-05-01): "3e" was added in audit-5 P1 #5 as a
# semantic alias for verification, but STAGE_DIRS has no "3e" entry → any
# code calling _version_path("3e") would KeyError. Removed the alias key;
# the verification stage uses the canonical "4" key everywhere (matches
# pipeline.py's existing write_stage_version("4") call sites).

STAGE_DIRS: Dict[str, Path] = {
    "1a": EVIDENCE_1A_DIR,
    "1b": EVIDENCE_1B_DIR,
    "2":  ROLLOUT_DIR,
    "3a": TASK_CARDS_DIR,
    "3b": PLACEMENTS_DIR,
    "3c": SAMPLES_3C_DIR,
    "4":  VERIFIED_DIR,
    "5":  FINAL_DIR,  # pass5 marker; rendered files live under DATA_ROOT/rendered/
}

# Downstream invalidation: changing stage X invalidates X and everything after.
# v12.11: "5" is post-verification render; downstream of "4".
PIPELINE_ORDER = ["1a", "1b", "2", "3a", "3b", "3c", "4", "5"]


def _version_path(stage: str) -> Path:
    return STAGE_DIRS[stage] / "_version"


def write_stage_version(stage: str) -> None:
    """Write the current version marker into the stage directory.

    Should be called by the pipeline once after the stage is fully
    completed for all videos in the batch.
    """
    if stage not in STAGE_VERSIONS:
        return
    d = STAGE_DIRS[stage]
    d.mkdir(parents=True, exist_ok=True)
    _version_path(stage).write_text(STAGE_VERSIONS[stage])


def stage_version_ok(stage: str) -> bool:
    """Return True if the stage's cache marker matches current version.

    If marker is missing (first run, or after rm -rf), treat as OK so
    the cache check itself doesn't force a rebuild — the per-video
    load_X functions will return None for missing files anyway.
    """
    if stage not in STAGE_VERSIONS:
        return True
    p = _version_path(stage)
    if not p.exists():
        # No marker → cache pre-versioning era, treat as needing rebuild
        # if any cached files exist for this stage; otherwise OK.
        d = STAGE_DIRS[stage]
        if not d.exists():
            return True
        existing = [f for f in d.iterdir() if f.suffix in (".json", ".jsonl")]
        return len(existing) == 0
    return p.read_text().strip() == STAGE_VERSIONS[stage]


def invalidate_stage_and_downstream(stage: str) -> None:
    """Delete cache files for `stage` and all stages after it.

    Used by --force_rerun_from. Removes generated data files while keeping
    directory structure. Stage 5 also owns rendered protocol variants, which
    live outside FINAL_DIR.
    """
    if stage not in PIPELINE_ORDER:
        raise ValueError(f"unknown stage: {stage}")
    start = PIPELINE_ORDER.index(stage)

    def _clear_generated_files(d: Path) -> None:
        if not d.exists():
            return
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.suffix in (".json", ".jsonl", ".parquet") or f.name == "_version":
                f.unlink()

    for s in PIPELINE_ORDER[start:]:
        d = STAGE_DIRS[s]
        _clear_generated_files(d)

    if "5" in PIPELINE_ORDER[start:]:
        for name in (
            "ts_image",
            "video_meta",
            "video_meta_standard_query_last",
            "trajectory",
        ):
            _clear_generated_files(DATA_ROOT / "rendered" / name)
