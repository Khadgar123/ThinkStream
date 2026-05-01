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
    EVIDENCE_1A_DIR,
    EVIDENCE_1B_DIR,
    ROLLOUT_DIR,
    TASK_CARDS_DIR,
    PLACEMENTS_DIR,
    SAMPLES_3C_DIR,
    VERIFIED_DIR,
)

# ---------------------------------------------------------------------------
# Per-stage version. Bump when you change the stage's prompt/code that
# would invalidate prior outputs. Downstream stages auto-invalidate.
# ---------------------------------------------------------------------------

STAGE_VERSIONS: Dict[str, str] = {
    # v12.11 audit-5 P1 #5 (2026-05-01): bumps below align with the v12.11
    # data-construction logic changes. Without these, an existing cluster
    # cache stamped v12.5 would silently reuse stale outputs:
    #   1a stays v12.5 — pass1a logic unchanged; current 320 outputs OK.
    #   1b → v12.11: compact prompt (5535555) + await fix (audit-4 P1#1) +
    #         singleton solo_<i> ids. Old v12.5 1b is functionally OK
    #         (verified by visual audit) but missing solo_ ids; bumping
    #         signals downstream that 1b-derived caches need re-derivation
    #         if you regenerate.
    #   2 stays v12.5 — pass2 logic unchanged.
    #   3a → v12.11: pass3a card targets bumped (N1 2→4, F6 1→2, F1 3→4),
    #         FAMILY_TARGETS shape changed → existing card pool stale.
    #   3b → v12.11: TIER_BONUS + answer_form bonus + PN1_PER_SEC
    #         duration-normalize + greedy diversity bumps.
    #   3c → v12.11: emit-every-chunk + render output-fallback fix.
    #   3e → v12.11: _is_v12_sample detector fix (no longer mistags v12).
    #   5  → v12.11: pass5 recall turn order + tools= for chat_template.
    "1a": "v12.5",
    "1b": "v12.11",
    "2":  "v12.5",
    "3a": "v12.11",
    "3b": "v12.11",
    "3c": "v12.11",
    "3e": "v12.11",
    "4":  "v12.5",  # legacy alias for 3e
    "5":  "v12.11",  # NEW — pass5_messages render version
}

STAGE_DIRS: Dict[str, Path] = {
    "1a": EVIDENCE_1A_DIR,
    "1b": EVIDENCE_1B_DIR,
    "2":  ROLLOUT_DIR,
    "3a": TASK_CARDS_DIR,
    "3b": PLACEMENTS_DIR,
    "3c": SAMPLES_3C_DIR,
    "4":  VERIFIED_DIR,
}

# Downstream invalidation: changing stage X invalidates X and everything after.
PIPELINE_ORDER = ["1a", "1b", "2", "3a", "3b", "3c", "4"]


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

    Used by --force_rerun_from. Removes only data files (json/jsonl),
    keeps directory structure.
    """
    if stage not in PIPELINE_ORDER:
        raise ValueError(f"unknown stage: {stage}")
    start = PIPELINE_ORDER.index(stage)
    for s in PIPELINE_ORDER[start:]:
        d = STAGE_DIRS[s]
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix in (".json", ".jsonl") or f.name == "_version":
                f.unlink()
