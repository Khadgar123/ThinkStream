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
    "1a": "v9.1",
    "1b": "v9.1",   # tightened state_changes fallback
    "2":  "v9.1",
    "3a": "v9.4.1", # v9.3: 8 family prompts MC-primary, N1 → MC, F5/F6
                    #       targets bumped, F6 verify range +8 forward.
                    # v9.4: 4 reasoning families CR1/CR2/CR3/CR4 added
                    #       (FAMILY_TARGETS / RETENTION_CLASS / FORCE_ATTEMPT
                    #       / FAMILY_PROMPTS / classify_chunks all touched).
                    # v9.4.1: HLD/N1 prompt corrected — gt is now literally
                    #       "Unable to answer" (matches OVO HLD 186/186
                    #       samples). Previous v9.4 had the inverse design
                    #       (gt = real visible option), which would teach the
                    #       wrong refusal direction at HLD eval.
    "3b": "v12.0",  # v12.0: bumped jointly with 3c/4 so the entire pass3
                    #       chain rebuilds under new code. pass3b_placement had
                    #       material changes in this commit (new visibility
                    #       checks, trajectory planning tweaks).
    "3c": "v12.0",  # v12.0: THINKSTREAM_PROTOCOL=v12 output format — tool_call +
                    #       answer tags instead of action/response. Bumped to
                    #       invalidate all v11-format cached samples.
    "4":  "v12.0", # v12.0: protocol-aware verification supporting v12
                    #       <tool_call>/<answer> format. v11-specific checks
                    #       (action/response tags) replaced with v12-equivalents.
                    #       Bumped jointly with 3c to force full rebuild.
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
