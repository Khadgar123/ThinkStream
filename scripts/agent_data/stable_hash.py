"""Stable hashing helpers for data construction.

Python's built-in ``hash()`` is randomized per interpreter process. Any data
construction decision that affects sampling, placement, split, or answer
balance must use these helpers instead so the same seed and inputs regenerate
the same corpus.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_int(*parts: Any, bits: int = 64) -> int:
    """Return a deterministic unsigned integer for ``parts``.

    ``bits`` is rounded down to a whole number of bytes and capped by SHA256's
    digest length. The default 64 bits is plenty for random seeds and modulo
    bucketing while keeping the resulting integer compact.
    """
    n_bytes = max(1, min(32, int(bits) // 8))
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:n_bytes], "big")


def stable_mod(*parts: Any, modulo: int) -> int:
    """Return ``stable_int(parts) % modulo`` with input validation."""
    if modulo <= 0:
        raise ValueError(f"modulo must be positive, got {modulo}")
    return stable_int(*parts) % int(modulo)


def stable_seed(seed: int, *parts: Any, modulo: int = 1_000_000) -> int:
    """Combine a human seed and stable hash parts into a Random seed."""
    return int(seed) + stable_mod(*parts, modulo=modulo)
