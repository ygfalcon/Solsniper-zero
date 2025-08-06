from __future__ import annotations

"""Simple progress logging utilities."""


_verbose: bool = True


def set_verbosity(verbose: bool) -> None:
    """Control whether :func:`step` outputs messages."""
    global _verbose
    _verbose = verbose


def step(message: str) -> None:
    """Print a progress message when verbosity is enabled."""
    if _verbose:
        print(message)
