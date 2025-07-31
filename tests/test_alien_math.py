import numpy as np
import pytest

from solhunter_zero.alien_math import decode_glyph_matrix, alien_prime_sequence, pyramid_frequency


def test_decode_glyph_matrix():
    matrix = [
        [0, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1],
    ]
    assert decode_glyph_matrix(matrix) == "hi"


def test_alien_prime_sequence():
    primes = alien_prime_sequence(5)
    assert primes.tolist() == [2, 3, 5, 7, 11]


def test_pyramid_frequency():
    freq = pyramid_frequency([1, 2, 3])
    assert freq == {1: 1, 2: 1, 3: 2, 5: 1, 8: 1}
