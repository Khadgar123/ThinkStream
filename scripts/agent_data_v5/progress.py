"""
Unified progress tracking for all pipeline stages.

Each stage gets a JSONL stream file that:
- Truncates on run start (fresh log per run)
- Appends one line per completed item (video or chunk) immediately
- Can be monitored with `tail -f`

Usage:
    tracker = ProgressTracker("pass1a", total=18000, audit_dir=AUDIT_DIR)
    await tracker.record(video_id="v1", chunk_idx=0, success=True, extra={...})
    tracker.summary()  # prints final stats
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Thread-safe progress counter + JSONL stream log."""

    def __init__(self, stage_name: str, total: int, audit_dir: Path):
        self.stage = stage_name
        self.total = total
        self.completed = 0
        self.success = 0
        self.failed = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()

        # JSONL stream file — truncate on init (new run)
        self.stream_path = audit_dir / f"{stage_name}_stream.jsonl"
        audit_dir.mkdir(parents=True, exist_ok=True)
        self.stream_path.write_text("")  # truncate
        logger.info(f"  [{stage_name}] progress → tail -f {self.stream_path}")

    async def record(self, success: bool, **extra):
        """Record one completed item. Appends to JSONL stream immediately."""
        async with self._lock:
            self.completed += 1
            if success:
                self.success += 1
            else:
                self.failed += 1
                # Log failures immediately
                vid = extra.get("video_id", "?")
                logger.warning(f"  [{self.stage}] FAILED: {vid} {extra}")

            # Append to JSONL stream
            entry = {
                "stage": self.stage,
                "success": success,
                "n": self.completed,
                **extra,
            }
            with open(self.stream_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Console progress every 20 items or on completion
            if self.completed % 20 == 0 or self.completed == self.total:
                elapsed = time.time() - self.start_time
                rps = self.completed / max(elapsed, 0.001)
                logger.info(
                    f"  [{self.stage}] {self.completed}/{self.total} "
                    f"(ok={self.success} fail={self.failed} {rps:.1f}/s)"
                )

    def record_sync(self, success: bool, **extra):
        """Synchronous version for non-async contexts."""
        self.completed += 1
        if success:
            self.success += 1
        else:
            self.failed += 1

        entry = {"stage": self.stage, "success": success, "n": self.completed, **extra}
        with open(self.stream_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.completed % 20 == 0 or self.completed == self.total:
            elapsed = time.time() - self.start_time
            rps = self.completed / max(elapsed, 0.001)
            logger.info(
                f"  [{self.stage}] {self.completed}/{self.total} "
                f"(ok={self.success} fail={self.failed} {rps:.1f}/s)"
            )

    def summary(self) -> Dict:
        """Return final stats dict."""
        elapsed = time.time() - self.start_time
        stats = {
            "stage": self.stage,
            "total": self.total,
            "completed": self.completed,
            "success": self.success,
            "failed": self.failed,
            "elapsed_sec": round(elapsed, 1),
            "rate_per_sec": round(self.completed / max(elapsed, 0.001), 2),
        }
        logger.info(
            f"  [{self.stage}] DONE: {self.success}/{self.completed} ok "
            f"in {elapsed:.0f}s ({stats['rate_per_sec']}/s)"
        )
        return stats
