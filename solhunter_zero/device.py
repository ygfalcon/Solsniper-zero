from __future__ import annotations

import os

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

__all__ = ["get_default_device", "DEFAULT_DEVICE"]

def get_default_device() -> str | "torch.device":
    """Return the preferred torch device.

    Checks the ``TORCH_DEVICE`` environment variable first. If not set, falls
    back to CUDA when available, then Apple's Metal Performance Shaders (MPS),
    otherwise CPU.
    """
    env_device = os.getenv("TORCH_DEVICE")
    if env_device:
        return torch.device(env_device) if torch is not None else env_device

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


DEFAULT_DEVICE = get_default_device()
