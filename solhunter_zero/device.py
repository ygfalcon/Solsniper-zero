from __future__ import annotations

import logging
import os

import torch


def get_default_device() -> str | None:
    """Return the default GPU backend if available.

    This checks for a usable CUDA or MPS device via PyTorch first and then
    falls back to CuPy. The function returns the backend name (``"torch"`` or
    ``"cupy"``) when a GPU is detected, otherwise ``None``.
    """

    try:  # pragma: no cover - optional dependency
        mps_available = (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available() or mps_available:
            if mps_available and os.environ.setdefault(
                "PYTORCH_ENABLE_MPS_FALLBACK", "1"
            ) != "1":
                logging.getLogger(__name__).warning(
                    "MPS is available but PYTORCH_ENABLE_MPS_FALLBACK is not set to '1'"
                )
            return "torch"
    except Exception:
        pass
    try:  # pragma: no cover - optional dependency
        import cupy as cp  # type: ignore

        if cp.cuda.runtime.getDeviceCount() > 0:
            return "cupy"
    except Exception:
        pass
    return None


def select_device(device: str | torch.device | None = "auto") -> torch.device:
    """Return a valid :class:`torch.device`.

    Parameters
    ----------
    device:
        Desired device identifier. If ``"auto"`` or ``None`` the function
        chooses ``"cuda"" when available, otherwise ``"cpu"". If a CUDA
        device is requested but unavailable the call falls back to the CPU.
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str) and device != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device) if isinstance(device, str) else device
