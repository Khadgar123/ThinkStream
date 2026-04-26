"""Reviewable training audit logger (SFT + GRPO).

Writes a JSONL stream per training stage so a run can be fully reconstructed
after the fact: every batch's loss, per-sample weights, action distribution,
and (for GRPO) every reward component before/after gating.

Files written to <audit_dir>/:
    sft_step.jsonl    one line per SFT optimization step
    sft_sample.jsonl  one line per sample seen (sample_id → loss, weight, type)
    grpo_step.jsonl   one line per GRPO step (aggregate reward stats)
    grpo_sample.jsonl one line per (sample, generation) with full reward dict

Each file truncates on init so a fresh run produces a clean log.
Tail-friendly: `tail -f audit/grpo_sample.jsonl | jq .` works during training.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """Best-effort rank check that doesn't require torch.distributed."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return os.environ.get("RANK", "0") == "0"


class AuditWriter:
    """Append-only JSONL writer with truncate-on-init and rank-0 gating.

    Thread-safe (uses a per-instance lock) so SFT/GRPO callbacks called from
    different threads don't interleave bytes within a record.
    """

    def __init__(self, path: Path, *, truncate: bool = True, rank0_only: bool = True):
        self.path = Path(path)
        self.enabled = (not rank0_only) or _is_rank0()
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                self.path.write_text("")
            logger.info("[audit] %s → tail -f %s", self.path.name, self.path)

    def write(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        record.setdefault("ts", time.time())
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", buffering=1) as f:
                f.write(line + "\n")


def resolve_audit_dir(
    explicit_dir: Optional[str],
    output_dir: Optional[str],
    fallback_subdir: str = "audit",
) -> Optional[Path]:
    """Pick an audit directory.

    Priority: explicit_dir > output_dir/audit. Returns None if neither given,
    in which case audit logging is disabled.
    """
    if explicit_dir:
        return Path(explicit_dir)
    if output_dir:
        return Path(output_dir) / fallback_subdir
    return None
