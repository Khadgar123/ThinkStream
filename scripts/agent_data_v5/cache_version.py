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
    #   1a stays v12.5 — pass1a logic unchanged; current 320 outputs OK.
    #   1b → v12.11: compact prompt (5535555) + await fix (audit-4 P1#1) +
    #         singleton solo_<i> ids. Old v12.5 1b is functionally OK
    #         (verified by visual audit) but missing solo_ ids; bumping
    #         signals downstream that 1b-derived caches need re-derivation
    #         if you regenerate.
    #   2 → v12.18: pass2 observation prompt/input now uses a text-first,
    #        forward-ordered timestamped image list over the full sliding
    #        visual window. This beat both chronological video blocks and
    #        reverse image order on the confirmed stale batchtest failures and
    #        avoids relying on OpenAI video_url timestamp behavior.
    #   3a stays v12.18: pass3 card generation is evidence/card based and does
    #        not consume pass2 rollout text.
    #   3b/3c/4/5 → v12.20: downstream trajectory caches consume pass2 rollout
    #        text/snapshots either directly or through generated samples, so they
    #        must be invalidated with the v12.18 pass2 prompt/input change.
    #   v12.18 background: pass3 display taxonomy fields, mixed MC answer
    #        protocols (letter/text/letter+text), semantic gold_answer split
    #        from SFT target, and verifier/rebalance updates.
    "1a": "v12.5",
    "1b": "v12.11",
    "2":  "v12.18",
    "3a": "v12.18",
    "3b": "v12.20",
    "3c": "v12.20",
    "4":  "v12.20",  # canonical key — verification
    "5":  "v12.20",  # pass5_messages render version
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
    "5":  FINAL_DIR,  # v12.11: pass5_messages writes *_messages.jsonl here
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
