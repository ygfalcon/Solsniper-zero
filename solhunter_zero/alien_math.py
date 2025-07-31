from __future__ import annotations

import numpy as np


def decode_glyph_matrix(matrix: list[list[int]] | np.ndarray) -> str:
    """Decode an Nx8 binary matrix into an ASCII string.

    Each row of ``matrix`` represents the bits of an ASCII character,
    most significant bit first.
    """
    arr = np.asarray(matrix, dtype=int)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError("matrix must be of shape (n, 8)")
    weights = 1 << np.arange(7, -1, -1)
    codes = arr.dot(weights)
    return "".join(map(chr, codes))


def alien_prime_sequence(n: int) -> np.ndarray:
    """Return the first ``n`` prime numbers as a NumPy array."""
    if n <= 0:
        return np.array([], dtype=int)

    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        arr = np.asarray(primes, dtype=int)
        if arr.size == 0 or np.all(candidate % arr != 0):
            primes.append(candidate)
        candidate += 1
    return np.array(primes, dtype=int)


def pyramid_frequency(values: list[int] | np.ndarray) -> dict[int, int]:
    """Build a summation pyramid from ``values`` and return value frequencies."""
    arr = np.asarray(values, dtype=int)
    if arr.size == 0:
        return {}

    levels = [arr]
    while arr.size > 1:
        arr = arr[:-1] + arr[1:]
        levels.append(arr)
    flat = np.concatenate(levels)
    uniq, counts = np.unique(flat, return_counts=True)
    return {int(u): int(c) for u, c in zip(uniq, counts)}
