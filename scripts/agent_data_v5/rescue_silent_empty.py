"""One-shot rescue for chunks marked _silent_empty whose stored _raw is parseable.

Why this exists: a previous version of pass1a's parse_evidence_result was missing
the markdown-fence strip, so chunks whose raw response was wrapped in ```json
... ``` were silently treated as empty. The current parser strips fences and
parses fine — running it on the stored _raw recovers content for ~99.7% of
silent_empty chunks. The remaining ~0.3% are real mid-output truncations and
are recovered by an array-level walker (see rescue_truncated_arrays).

Idempotent: chunks already restored (no _silent_empty / no _raw) are skipped.

Run:
    python -m scripts.agent_data_v5.rescue_silent_empty
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import EVIDENCE_1A_DIR, EVIDENCE_1B_DIR
from .pass1a_evidence import parse_evidence_result

logger = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.removeprefix("```json").removeprefix("```").strip()
    raw = raw.removesuffix("```").strip()
    return raw


def _walk_object_array(s: str, start: int) -> List[Dict]:
    """Greedy element-by-element parse of `[{...}, {...}, ...`. Tolerates a
    truncated tail by stopping at the first incomplete element."""
    items: List[Dict] = []
    i = start
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t\n\r,":
            i += 1
        if i >= n or s[i] == "]":
            break
        if s[i] != "{":
            break
        depth = 0
        j = i
        in_str = False
        esc = False
        closed = False
        while j < n:
            c = s[j]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif in_str:
                if c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            items.append(json.loads(s[i : j + 1]))
                        except (json.JSONDecodeError, ValueError):
                            pass
                        closed = True
                        j += 1
                        break
            j += 1
        if not closed:
            break
        i = j
    return items


def _walk_string_array(s: str, start: int) -> List[str]:
    items: List[str] = []
    i = start
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t\n\r,":
            i += 1
        if i >= n or s[i] == "]":
            break
        if s[i] != '"':
            break
        j = i + 1
        esc = False
        closed = False
        while j < n:
            if esc:
                esc = False
            elif s[j] == "\\":
                esc = True
            elif s[j] == '"':
                closed = True
                break
            j += 1
        if not closed:
            break
        try:
            items.append(json.loads(s[i : j + 1]))
        except (json.JSONDecodeError, ValueError):
            pass
        i = j + 1
    return items


def _walk_string_field(s: str, start: int) -> str:
    """Parse a JSON string starting at index `start` (which points to opening
    quote)."""
    if start >= len(s) or s[start] != '"':
        return ""
    j = start + 1
    esc = False
    while j < len(s):
        if esc:
            esc = False
        elif s[j] == "\\":
            esc = True
        elif s[j] == '"':
            try:
                return json.loads(s[start : j + 1])
            except (json.JSONDecodeError, ValueError):
                return ""
        j += 1
    return ""


def rescue_truncated_arrays(raw: str) -> Optional[Dict]:
    """Element-walk rescue for truncated JSON. Returns a dict shaped like the
    parser output if any field is non-empty, else None.

    Used as a last-resort path when both json.loads and the brace-balance
    fallback fail (typical: response cut mid-element by max_tokens cap)."""
    s = _strip_fences(raw)
    out: Dict = {"visible_entities": [], "atomic_facts": [], "ocr": [], "spatial": ""}

    m = re.search(r'"visible_entities"\s*:\s*\[', s)
    if m:
        out["visible_entities"] = _walk_object_array(s, m.end())

    m = re.search(r'"atomic_facts"\s*:\s*\[', s)
    if m:
        # Could be list[str] (v9.5) or list[dict] (legacy). Try object walk
        # first; if zero items, retry as string walk.
        objs = _walk_object_array(s, m.end())
        if objs:
            out["atomic_facts"] = objs
        else:
            out["atomic_facts"] = _walk_string_array(s, m.end())

    m = re.search(r'"ocr"\s*:\s*\[', s)
    if m:
        out["ocr"] = _walk_string_array(s, m.end())

    m = re.search(r'"spatial"\s*:\s*', s)
    if m:
        out["spatial"] = _walk_string_field(s, m.end())

    if (
        out["visible_entities"]
        or out["atomic_facts"]
        or out["ocr"]
        or out["spatial"].strip()
    ):
        return out
    return None


def _rescue_one_chunk(chunk: Dict) -> Tuple[bool, str]:
    """Returns (recovered, mode). mode ∈ {'parse', 'walker', 'noop', 'skip'}.

    A chunk is recovered if its visible_entities/atomic_facts/ocr/spatial
    were empty AND the stored _raw yields content via standard parse OR the
    array walker. Mutates chunk in place when recovered.
    """
    if not chunk.get("_silent_empty"):
        return False, "skip"
    raw = chunk.get("_raw")
    if not raw:
        return False, "skip"

    # Path 1: standard parse (covers ~99.7% — old _raw lacked fence-strip).
    parsed = parse_evidence_result(raw, {"time": chunk.get("time", [0, 0])})
    if parsed.get("parse_success"):
        chunk["visible_entities"] = parsed["visible_entities"]
        chunk["atomic_facts"] = parsed["atomic_facts"]
        chunk["ocr"] = parsed["ocr"]
        chunk["spatial"] = parsed["spatial"]
        chunk["parse_success"] = True
        chunk["_rescued"] = "parse"
        for k in ("_silent_empty", "_raw", "_retry_failed"):
            chunk.pop(k, None)
        return True, "parse"

    # Path 2: element-walker (covers truncated mid-output JSON).
    walked = rescue_truncated_arrays(raw)
    if walked is not None:
        from .pass1a_evidence import _normalize_entity, _normalize_atomic_fact

        chunk["visible_entities"] = [_normalize_entity(e) for e in walked["visible_entities"]]
        chunk["atomic_facts"] = [_normalize_atomic_fact(f) for f in walked["atomic_facts"]]
        chunk["ocr"] = walked["ocr"]
        chunk["spatial"] = walked["spatial"]
        chunk["parse_success"] = True
        chunk["_rescued"] = "walker"
        for k in ("_silent_empty", "_raw", "_retry_failed"):
            chunk.pop(k, None)
        return True, "walker"

    return False, "noop"


def rescue_directory(evidence_dir: Path) -> Dict[str, int]:
    stats = {"files": 0, "chunks": 0, "silent_empty": 0,
             "rescued_parse": 0, "rescued_walker": 0, "still_failing": 0}
    for path in sorted(evidence_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"  skip unreadable {path.name}: {e}")
            continue
        if not isinstance(data, list):
            continue
        stats["files"] += 1
        stats["chunks"] += len(data)
        file_changed = False
        for chunk in data:
            if not isinstance(chunk, dict):
                continue
            if chunk.get("_silent_empty"):
                stats["silent_empty"] += 1
                ok, mode = _rescue_one_chunk(chunk)
                if ok and mode == "parse":
                    stats["rescued_parse"] += 1
                    file_changed = True
                elif ok and mode == "walker":
                    stats["rescued_walker"] += 1
                    file_changed = True
                elif not ok:
                    stats["still_failing"] += 1
        if file_changed:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for label, d in [("evidence_1a", EVIDENCE_1A_DIR), ("evidence_1b", EVIDENCE_1B_DIR)]:
        if not d.exists():
            print(f"{label}: directory missing, skipping")
            continue
        print(f"\n=== {label}: {d} ===")
        s = rescue_directory(d)
        print(f"  files scanned        : {s['files']}")
        print(f"  total chunks         : {s['chunks']}")
        print(f"  silent_empty found   : {s['silent_empty']}")
        print(f"  rescued via parse    : {s['rescued_parse']}")
        print(f"  rescued via walker   : {s['rescued_walker']}")
        print(f"  still failing        : {s['still_failing']}")
    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main() or 0)
