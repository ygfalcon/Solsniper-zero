from __future__ import annotations

import math
from typing import Iterable, List


def decode_glyph(text: str) -> List[int]:
    """Decode ``text`` into a sequence of numeric glyphs.

    Each character is mapped to a small integer so that the function
    works deterministically across environments.
    """
    return [ord(ch) % 16 for ch in text]


def conviction_score(
    glyphs: Iterable[int], *, depth: float | None = None, imbalance: float | None = None
) -> float:
    """Return a conviction score based on decoded glyphs.

    Parameters
    ----------
    glyphs:
        Sequence of integers returned by :func:`decode_glyph`.
    depth:
        Optional order book depth value.
    imbalance:
        Optional order book imbalance value.
    """
    vals = list(glyphs)
    if not vals:
        return 0.0
    base = sum(vals) / len(vals)
    if depth is not None:
        base += float(depth) * 0.1
    if imbalance is not None:
        base -= float(imbalance) * 0.05
    # Compress to [-1, 1]
    return math.tanh(base / 10.0)
