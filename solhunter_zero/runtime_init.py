"""Runtime initialization helpers for SolHunter Zero.

This module centralizes GPU and thread setup so that dependent modules can
rely on a single source of truth.  Each helper runs at most once per process.
"""
from __future__ import annotations

from . import device, system

_GPU_ENV: dict[str, str] | None = None
_RAYON_SET = False


def initialize_gpu() -> dict[str, str]:
    """Initialize GPU environment variables once and return them."""
    global _GPU_ENV
    if _GPU_ENV is None:
        _GPU_ENV = device.initialize_gpu()
    return _GPU_ENV


def set_rayon_threads() -> None:
    """Configure Rayon thread count once for Rust FFI bindings."""
    global _RAYON_SET
    if not _RAYON_SET:
        system.set_rayon_threads()
        _RAYON_SET = True


__all__ = ["initialize_gpu", "set_rayon_threads"]
