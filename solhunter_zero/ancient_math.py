"""Numerical helpers inspired by ancient alien artifacts."""

from __future__ import annotations

from typing import Sequence


__all__ = ["decode_glyph_series", "pyramid_transform"]


def decode_glyph_series(data: Sequence[int]) -> list[float]:
    """Decode a sequence of glyph integers into normalized values.

    The algorithm is based on patterns etched into theoretical alien relics.
    Each integer is centered around the series median and scaled to the
    range [-1, 1].

    Examples
    --------
    >>> decode_glyph_series([1, 2, 3])
    [-1.0, 0.0, 1.0]
    """

    values = [float(x) for x in data]
    if not values:
        return []
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = (
        ordered[mid]
        if len(ordered) % 2
        else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    centered = [x - median for x in values]
    max_abs = max(abs(x) for x in centered) or 1.0
    return [x / max_abs for x in centered]


def pyramid_transform(x: float, level: int) -> float:
    """Apply a pyramidal scaling reminiscent of extraterrestrial geometry.

    Multiplying by the golden ratio ``level`` times echoes the tiered
    construction of hypothetical alien pyramids.

    Examples
    --------
    >>> pyramid_transform(1.0, 2)
    2.618033988749895
    """

    phi = (1 + 5**0.5) / 2
    result = float(x)
    for _ in range(max(0, level)):
        result *= phi
    return result
