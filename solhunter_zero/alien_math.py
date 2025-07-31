from __future__ import annotations
from typing import Sequence


def glyph_strength(pattern: Sequence[int]) -> float:
    """Return average glyph value."""
    if not pattern:
        return 0.0
    return sum(pattern) / len(pattern)


def glyph_bias(pattern: Sequence[int]) -> float:
    """Return a simple directional bias of the glyphs."""
    if not pattern:
        return 0.0
    center = (len(pattern) - 1) / 2.0
    return sum((i - center) * v for i, v in enumerate(pattern)) / len(pattern)
