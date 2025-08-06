from __future__ import annotations

import torch


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


def get_default_device() -> torch.device:
    """Return the preferred accelerator device if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - optional backend
        return torch.device("mps")
    return torch.device("cpu")
